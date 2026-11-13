"""Upgrading a node of a graph record from the version it was saved at to a later one.

A node definition's ``Upgrade`` steps are checked when it is declared. ``registry.upgrade`` applies them to one node
of a graph record: what each input gets (a constant, a graph input or edges), its version, and the edges out
of an output it renames. ``Graph.compile`` never upgrades: a node pinned at an old version runs that version.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import (
    Address,
    Graph,
    GraphRecord,
    InputEntry,
    InvalidCallError,
    NodeDeclarationError,
    NodeRecord,
    NodeRegistry,
    Param,
    Upgrade,
    host_type,
    make_graph,
    node,
    subgraph,
)
from conductor.widgets import Textarea

from tests.test_core.nodes import registry_of


@host_type("upgrade-test-txt", title="Text")
class Txt(str):
    pass


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

@node("rename", title="Rename", description="d", category="test", replaces=node("rename", title="Rename v2", description="For a test.", replaces=node("rename", title="Rename v1", description="For a test.")(rename_v1), upgrade=Upgrade(rewrite=rename_upgrade_1))(rename_v2), upgrade=Upgrade(renamed_outputs={"result": "text"}))
def rename(
    first: Annotated[Txt, _param("First")] = Txt(""),
    last: Annotated[Txt, _param("Last")] = Txt(""),
) -> Named:
    """Version 1 took ``name``; 2 splits it into ``first`` and ``last``; 3 renames the output ``result`` to ``text``."""
    return Named(text=Txt(f"{first} {last}"))


@node("echo", title="Echo", description="d", category="test")
def echo(text: Annotated[Txt, _param("Text")] = Txt("")) -> Annotated[Txt, Param(title="R")]:
    return text


def _registry() -> NodeRegistry:
    registry = registry_of(rename, echo)
    return registry


def _graph(version: int = 1) -> GraphRecord:
    return GraphRecord(
        nodes={
            "r": NodeRecord(node="rename", version=version, inputs={"name": InputEntry(constant="Ada Lovelace")} if version == 1 else {}),
            "e": NodeRecord(node="echo", inputs={"text": InputEntry(sources=(Address("r", "result" if version < 3 else "text"),))}),
        },
        outputs={"result": Address("e", "result")},
    )


def _inputs(record: GraphRecord, node_id: str) -> dict:
    """What each input of the node ``node_id`` gets, as the record's JSON writes it."""
    return record.model_dump(mode="json")["nodes"][node_id].get("inputs", {})


# --- an input or output the next version drops needs an upgrade ------------------


def _older_x(x: Annotated[Txt, _param("X")]) -> Annotated[Txt, Param(title="R")]:
    return x


def _newer_y(y: Annotated[Txt, _param("Y")]) -> Annotated[Txt, Param(title="R")]:
    return y


def test_an_input_gone_from_the_next_version_needs_an_upgrade():
    with pytest.raises(NodeDeclarationError) as raised:
        node("two", title="Newer y", description="For a test.", replaces=node("two", title="Older x", description="For a test.")(_older_x))(_newer_y)
    assert raised.value.problems[0].type == "upgrade_missing" and "two" in str(raised.value)


def test_an_upgrade_must_say_where_a_gone_input_went():
    with pytest.raises(NodeDeclarationError, match="upgrade|Upgrade"):
        node("two", title="Newer y", description="For a test.", replaces=node("two", title="Older x", description="For a test.")(_older_x), upgrade=Upgrade(renamed_outputs={}))(_newer_y)


def test_a_rename_or_a_rewrite_says_where_it_went():
    assert node("two", title="Newer y", description="For a test.", replaces=node("two", title="Older x", description="For a test.")(_older_x), upgrade=Upgrade(renamed_inputs={"x": "y"}))(_newer_y).version == 2
    assert node("two", title="Newer y", description="For a test.", replaces=node("two", title="Older x", description="For a test.")(_older_x), upgrade=Upgrade(rewrite=lambda v: {"y": v["x"]}))(_newer_y).version == 2


def test_a_version_keeping_its_fields_needs_no_upgrade():
    assert node("two", title="Older x", description="For a test.", replaces=node("two", title="Older x", description="For a test.")(_older_x))(_older_x).version == 2


# --- registry.upgrade ------------------------------------------------------------


def test_a_two_step_upgrade_rewrites_the_constants_and_sets_the_last_version():
    upgraded = _registry().upgrade(_graph(), "r")

    assert upgraded.nodes["r"].version == 3
    assert _inputs(upgraded, "r") == {"first": {"constant": "Ada"}, "last": {"constant": "Lovelace"}}


def test_a_renamed_output_moves_the_edges_out_of_it_and_the_graph_compiles_clean():
    registry = _registry()
    upgraded = registry.upgrade(_graph(), "r")

    assert _inputs(upgraded, "e") == {"text": {"sources": [{"node": "r", "output": "text"}]}}
    graph = Graph.compile(upgraded, registry)
    assert graph.is_runnable and graph.problems == [], graph.problems


def test_upgrade_with_to_stops_at_that_version():
    upgraded = _registry().upgrade(_graph(), "r", to=2)

    assert upgraded.nodes["r"].version == 2
    assert _inputs(upgraded, "e") == {"text": {"sources": [{"node": "r", "output": "result"}]}}


def test_upgrade_reads_the_record_through_the_record_door_and_refuses_two_nodes_of_one_id_or_an_id_that_is_no_name():
    two_of_one_id = '{"nodes": {"r": {"node": "rename"}, "r": {"node": "rename", "version": 2}}}'
    id_no_name = '{"nodes": {"a.b": {"node": "rename"}}}'

    with pytest.raises(InvalidCallError) as refused:
        _registry().upgrade(two_of_one_id, "r")
    assert [(p.type, p.ctx["error"]) for p in refused.value.problems] == [("invalid_record", "repeated_key")]
    with pytest.raises(InvalidCallError) as refused:
        _registry().upgrade(id_no_name, "a.b")
    assert [(p.type, p.loc) for p in refused.value.problems] == [("invalid_record", ("record", "nodes", "a.b", "[key]"))]
    with pytest.raises(InvalidCallError) as refused:
        _registry().upgrade(_graph(), "nope")
    assert [(p.type, p.ctx["nodes"]) for p in refused.value.problems] == [("node_not_in_graph", ["e", "r"])]


def test_a_node_already_at_its_latest_version_returns_the_graph_unchanged():
    graph = _graph(version=3)
    assert _registry().upgrade(graph, "r") is graph


def test_an_edge_into_an_input_the_step_renames_moves_with_it():
    def carry_v1(name: Annotated[Txt, _param("Name")] = Txt("")) -> Annotated[Txt, Param(title="R")]:
        return name

    @node("carry", title="Carry", description="d", category="test", replaces=node("carry", title="Carry v1", description="For a test.")(carry_v1), upgrade=Upgrade(renamed_inputs={"name": "full"}))
    def carry(full: Annotated[Txt, _param("Full")] = Txt("")) -> Annotated[Txt, Param(title="R")]:
        return full

    registry = registry_of(echo, carry)
    graph = GraphRecord.model_validate({"nodes": {"src": {"node": "echo"}, "r": {"node": "carry", "inputs": {"name": {"sources": [{"node": "src", "output": "result"}]}}}}})

    upgraded = registry.upgrade(graph, "r")
    assert _inputs(upgraded, "r") == {"full": {"sources": [{"node": "src", "output": "result"}]}}


def _versions_of_summed(kind: str, upgrade: Upgrade | None):
    """Two versions of the node ``summed``, as functions or as subgraphs: the first takes ``a`` and ``b``, the
    second only ``a``. Returns the newer version and the nodes the subgraphs use."""

    @node("join", title="Join", description="For a test.")
    def join(a: int, b: int) -> int:
        return a + b

    @node("one", title="One", description="For a test.")
    def one(a: int) -> int:
        return a

    if kind == "function":
        older = node("summed", title="Summed v1", description="For a test.")(join.function)
        return node("summed", title="Summed", description="For a test.", replaces=older, upgrade=upgrade)(one.function), []
    older = subgraph("summed", title="Summed v1", description="For a test.")(make_graph(join))
    return subgraph("summed", title="Summed", description="For a test.", replaces=older, upgrade=upgrade)(make_graph(one)), [join, one]


_KINDS = pytest.mark.parametrize("kind", ["function", "subgraph"])


@_KINDS
def test_a_version_that_drops_an_input_needs_an_upgrade_for_a_function_and_a_subgraph_alike(kind: str) -> None:
    with pytest.raises(NodeDeclarationError) as raised:
        _versions_of_summed(kind, None)
    assert raised.value.problems[0].type == "upgrade_missing" and "summed" in str(raised.value)


@_KINDS
@pytest.mark.parametrize("gone", [
    {"constant": 2}, {"graph_input": "bee"}, {"sources": [{"node": "t", "output": "result"}]},
], ids=["constant", "graph-input", "edge"])
def test_an_input_the_upgrade_removes_loses_its_constant_its_graph_input_or_its_edges(kind: str, gone: dict) -> None:
    @node("three", title="Three", description="For a test.")
    def three() -> int:
        return 3

    summed, used = _versions_of_summed(kind, Upgrade(rewrite=lambda before: {k: v for k, v in before.items() if k != "b"}))
    registry = registry_of(three, summed, *used)
    graph = GraphRecord.model_validate({"nodes": {"t": {"node": "three"}, "s": {"node": "summed", "version": 1, "inputs": {"a": {"constant": 1}, "b": gone}}}})
    assert Graph.compile(graph, registry).problems == []

    upgraded = registry.upgrade(graph, "s")
    assert _inputs(upgraded, "s") == {"a": {"constant": 1}}
    assert Graph.compile(upgraded, registry).problems == []


def test_a_rewrite_sees_constants_and_defaults_together_and_a_default_stays_a_default():
    def counted_v1(src: Annotated[Txt, _param("Source")] = Txt(""), n: int = 0) -> Annotated[Txt, Param(title="R")]:
        return src

    @node("counted", title="Counted", description="For a test.", replaces=node("counted", title="Counted v1", description="For a test.")(counted_v1), upgrade=Upgrade(rewrite=lambda before: {**before, "n": before.get("n", 0) + 1, "added": 1}))
    def counted(src: Annotated[Txt, _param("Source")] = Txt(""), n: int = 0, added: int = 0) -> Annotated[Txt, Param(title="R")]:
        return src

    registry = registry_of(counted)
    with_both = registry.upgrade(GraphRecord.model_validate({"nodes": {"c": {"node": "counted", "inputs": {"n": {"constant": 2}, "src": {"graph_input": "src", "default": "x"}}}}}), "c")
    with_neither = registry.upgrade(GraphRecord.model_validate({"nodes": {"c": {"node": "counted"}}}), "c")

    assert _inputs(with_both, "c") == {"n": {"constant": 3}, "added": {"constant": 1}, "src": {"graph_input": "src", "default": "x"}}
    assert _inputs(with_neither, "c") == {"n": {"constant": 1}, "added": {"constant": 1}}


def test_an_edge_reading_a_field_of_a_renamed_output_moves_with_it():
    registry = _registry()
    graph = GraphRecord.model_validate({"nodes": {"r": {"node": "rename", "version": 2}, "e": {"node": "echo", "inputs": {"text": {"sources": [{"node": "r", "output": "result", "field": "text"}]}}}}})

    assert _inputs(registry.upgrade(graph, "r"), "e") == {"text": {"sources": [{"node": "r", "output": "text", "field": "text"}]}}


def test_a_downgrade_or_an_unknown_node_raises():
    registry = _registry()
    with pytest.raises(InvalidCallError, match="cannot move from version 3 to 1"):
        registry.upgrade(_graph(version=3), "r", to=1)
    with pytest.raises(InvalidCallError, match="nope"):
        registry.upgrade(_graph(), "nope")


def test_compile_leaves_a_pinned_version_alone():
    registry = _registry()
    graph = Graph.compile(_graph(), registry)

    assert graph.describe().nodes["r"].version == 1
    assert graph.is_runnable, graph.problems
    assert graph.run().outputs == {"result": "Ada Lovelace"}


# --- what moves with a renamed field ---------------------------------------------


def renames_v1(name: Annotated[Txt, _param("Name")] = Txt("")) -> Annotated[Txt, Param(title="R")]:
    return name

@node("renames", title="Renames", description="d", category="test", replaces=node("renames", title="Renames v1", description="For a test.")(renames_v1), upgrade=Upgrade(renamed_inputs={"name": "full"}, renamed_outputs={"result": "text"}))
def renames(full: Annotated[Txt, _param("Full")] = Txt("")) -> Named:
    """Version 2 renames the input ``name`` to ``full`` and the output ``result`` to ``text``."""
    return Named(text=full)


def test_a_renamed_input_takes_its_value_with_it():
    registry = registry_of(echo, renames)
    graph = GraphRecord.model_validate({"nodes": {"r": {"node": "renames", "inputs": {"name": {"constant": "Ada"}}}, "e": {"node": "echo", "inputs": {"text": {"sources": [{"node": "r", "output": "result"}]}}}}})

    upgraded = registry.upgrade(graph, "r")

    assert _inputs(upgraded, "r") == {"full": {"constant": "Ada"}}
    assert _inputs(upgraded, "e") == {"text": {"sources": [{"node": "r", "output": "text"}]}}
    assert Graph.compile(upgraded, registry).problems == []


def test_an_edge_follows_its_output_when_a_later_step_reuses_the_outputs_old_name():
    """Step 1 renames ``result`` to ``first``; step 2 declares an output named ``result`` again and renames it to
    ``second``. An edge saved at version 1 read the output that version 3 calls ``first``."""

    def reuse_outputs():
        return {}

    def reuse_v1() -> Annotated[Txt, Param(title="R")]:
        return Txt("")

    def reuse_v2() -> Mapping[str, Txt]:
        return {}

    @node("reuse", title="Reuse", description="d", category="test", compute_outputs=reuse_outputs, replaces=node("reuse", title="Reuse v2", description="For a test.", compute_outputs=reuse_outputs, replaces=node("reuse", title="Reuse v1", description="For a test.")(reuse_v1), upgrade=Upgrade(renamed_outputs={"result": "first"}))(reuse_v2), upgrade=Upgrade(renamed_outputs={"result": "second"}))
    def reuse() -> Mapping[str, Txt]:
        return {}

    registry = registry_of(echo, reuse)
    graph = GraphRecord.model_validate({"nodes": {"u": {"node": "reuse"}, "e": {"node": "echo", "inputs": {"text": {"sources": [{"node": "u", "output": "result"}]}}}}})

    upgraded = registry.upgrade(graph, "u")

    assert _inputs(upgraded, "e") == {"text": {"sources": [{"node": "u", "output": "first"}]}}


def test_upgrading_a_subgraph_sets_only_its_version():
    """A subgraph whose later version takes the same graph inputs, with no ``Upgrade`` to run: the node gets the later
    version and keeps its inputs as saved."""

    def inner_graph(inner_id: str) -> GraphRecord:
        return GraphRecord(nodes={inner_id: NodeRecord(node="echo", inputs={"text": InputEntry(graph_input="text")})})

    first = subgraph("inner_graph", title="Sub", description="d")(inner_graph("inner"))
    registry = registry_of(echo, subgraph("inner_graph", title="Sub", description="d", replaces=first)(inner_graph("renamed")))
    graph = GraphRecord.model_validate({"nodes": {"f": {"node": "inner_graph", "inputs": {"text": {"constant": "x"}}}}})

    upgraded = registry.upgrade(graph, "f")

    assert upgraded.nodes["f"].version == 2
    assert _inputs(upgraded, "f") == {"text": {"constant": "x"}}
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
def test_an_upgrade_that_drops_an_output_without_renaming_it_is_refused_where_it_is_declared(upgrade: Upgrade) -> None:
    """A rewrite reads the node's constants, never the edges out of it, so it cannot say where an output went."""
    with pytest.raises(NodeDeclarationError, match="'x'") as raised:
        node("o", title="Newer out", description="For a test.", replaces=node("o", title="Older out", description="For a test.")(_older_out), upgrade=upgrade)(_newer_out)
    assert raised.value.problems[0].type == "upgrade_missing"


def test_an_output_an_upgrade_renames_needs_nothing_more():
    assert node("o", title="Newer out", description="For a test.", replaces=node("o", title="Older out", description="For a test.")(_older_out), upgrade=Upgrade(renamed_outputs={"x": "y"}))(_newer_out).version == 2


def test_a_rewrite_returning_a_set_is_refused_as_set_has_no_order_at_the_node():
    def tagged_v1(tag: str) -> str:
        return tag

    @node("tagged", title="Tagged", description="For a test.", replaces=node("tagged", title="Tagged v1", description="For a test.")(tagged_v1), upgrade=Upgrade(rewrite=lambda before: {"tags": set(before.values())}))
    def tagged(tags: list[str]) -> str:
        return ",".join(tags)

    with pytest.raises(InvalidCallError) as refused:
        registry_of(tagged).upgrade(GraphRecord.model_validate({"nodes": {"t": {"node": "tagged", "inputs": {"tag": {"constant": "a"}}}}}), "t")
    (problem,) = refused.value.problems
    assert (problem.type, problem.loc, [e["type"] for e in problem.ctx["errors"]]) == ("invalid_value", ("t", "tags"), ["set_has_no_order"])


def test_a_rewrite_is_written_as_json_and_one_with_no_json_form_is_invalid_value_at_the_node():
    import datetime

    import polars as pl

    @node("when", title="When v1", description="For a test.")
    def when_v1(day: str) -> str:
        return day

    saved = GraphRecord.model_validate({"nodes": {"w": {"node": "when", "inputs": {"day": {"constant": "2026-01-02"}}}}})
    for returned, written in ((datetime.date(2026, 1, 2), "2026-01-02"), (pl.DataFrame({"a": [1]}), None)):

        @node("when", title="When", description="For a test.", replaces=when_v1, upgrade=Upgrade(rewrite=lambda before, r=returned: {"day": r}))
        def when(day: datetime.date) -> str:
            return day.isoformat()

        registry = NodeRegistry()
        registry.add(when)
        if written is None:
            with pytest.raises(InvalidCallError) as refused:
                registry.upgrade(saved, "w")
            assert [(p.type, p.loc) for p in refused.value.problems] == [("invalid_value", ("w", "day"))]
        else:
            upgraded = registry.upgrade(saved, "w")
            assert _inputs(upgraded, "w") == {"day": {"constant": written}}
            assert GraphRecord.model_validate_json(upgraded.model_dump_json()) == upgraded


def test_a_subgraphs_input_rename_moves_only_the_edge_into_that_exact_input():
    """The subgraph's inputs are ``a``, ``ab`` and ``cb``: renaming ``a`` leaves the edges into ``ab`` and ``cb``
    where they are."""

    @node("takes-b", title="Takes b", description="For a test.")
    def takes_b(b: str) -> str:
        return b

    @node("takes-a", title="Takes a", description="For a test.")
    def takes_a(a: str) -> str:
        return a

    @node("takes-z", title="Takes z", description="For a test.")
    def takes_z(z: str) -> str:
        return z

    @node("word", title="Word", description="For a test.")
    def word() -> str:
        return "w"

    def inner(last: str, name: str) -> GraphRecord:
        return GraphRecord.model_validate({"nodes": {
            "a": {"node": "takes-b", "inputs": {"b": {"graph_input": "ab"}}},
            "c": {"node": "takes-b", "inputs": {"b": {"graph_input": "cb"}}},
            "x": {"node": last, "inputs": {name: {"graph_input": name}}},
        }})

    v1 = subgraph("sub", title="Sub", description="For a test.")(inner("takes-a", "a"))
    v2 = subgraph("sub", title="Sub", description="For a test.", replaces=v1, upgrade=Upgrade(renamed_inputs={"a": "z"}))(inner("takes-z", "z"))
    reg = registry_of(takes_b, takes_a, takes_z, word, v2)
    word_result = {"sources": [{"node": "w", "output": "result"}]}
    stored = GraphRecord.model_validate({"nodes": {"w": {"node": "word"}, "s": {"node": "sub", "version": 1, "inputs": {"ab": word_result, "cb": word_result, "a": word_result}}}})
    assert Graph.compile(stored, reg).problems == []

    moved = reg.upgrade(stored, "s")

    assert _inputs(moved, "s") == {"ab": word_result, "cb": word_result, "z": word_result}
    assert Graph.compile(moved, reg).problems == []


def test_a_rename_carries_a_constant_a_graph_input_and_edges_as_they_are():
    def carry_v1(a: str, b: str, c: str) -> str:
        return a + b + c

    @node("carry3", title="Carry", description="For a test.", replaces=node("carry3", title="Carry v1", description="For a test.")(carry_v1), upgrade=Upgrade(renamed_inputs={"a": "x", "b": "y", "c": "z"}))
    def carry(x: str, y: str, z: str) -> str:
        return x + y + z

    registry = registry_of(echo, carry)
    graph = GraphRecord.model_validate({"nodes": {"src": {"node": "echo"}, "k": {"node": "carry3", "inputs": {
        "a": {"constant": "1"}, "b": {"graph_input": "bee", "default": "2"}, "c": {"sources": [{"node": "src", "output": "result"}]},
    }}}, "outputs": {"result": {"node": "k", "output": "result"}}})

    upgraded = registry.upgrade(graph, "k")

    assert _inputs(upgraded, "k") == {"x": {"constant": "1"}, "y": {"graph_input": "bee", "default": "2"}, "z": {"sources": [{"node": "src", "output": "result"}]}}
    assert Graph.compile(upgraded, registry).problems == []


def test_a_rewrite_giving_a_value_to_an_input_with_edges_is_refused():
    """An input gets one of a constant, a graph input or edges: a rewrite that gives a value to an input edges lead
    into would give it two, so the record door refuses what the step wrote."""

    def fed_v1(text: str) -> str:
        return text

    @node("fed", title="Fed", description="For a test.", replaces=node("fed", title="Fed v1", description="For a test.")(fed_v1), upgrade=Upgrade(rewrite=lambda before: {**before, "text": "x"}))
    def fed(text: str) -> str:
        return text

    registry = registry_of(echo, fed)
    graph = GraphRecord.model_validate({"nodes": {"src": {"node": "echo"}, "f": {"node": "fed", "inputs": {"text": {"sources": [{"node": "src", "output": "result"}]}}}}})

    with pytest.raises(InvalidCallError) as refused:
        registry.upgrade(graph, "f")
    [problem] = refused.value.problems
    assert (problem.type, problem.loc, problem.ctx["error"], problem.ctx["written_by"]) == (
        "invalid_record", ("record", "nodes", "f", "inputs", "text"), "one_of_constant_graph_input_sources", "the Upgrade of 'fed' to version 2",
    )
    assert problem.msg.endswith("Fix the Upgrade of 'fed' to version 2.")
