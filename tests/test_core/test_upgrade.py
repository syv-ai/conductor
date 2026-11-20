"""Tests for upgrading a node in a graph record to a later version of it.

A graph record says which version of each node it was saved with. A node's later version can say how to move a
record forward, with an ``Upgrade`` step. Those steps are checked when the node is declared.

``registry.upgrade`` applies the steps to one node of a record. It updates what each input gets (a constant, a graph
input or edges), the node's version, and the edges out of any output it renames. ``Graph.compile`` never upgrades on
its own: a node pinned at an old version runs that version.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import (
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    NodeRecord,
    NodeRegistry,
    Param,
    Source,
    Upgrade,
    host_type,
    make_graph,
    node,
    subgraph,
)
from conductor.widgets import Textarea

from tests.test_core.nodes import input_entry, registry_of


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


@node(
    "rename",
    title="Rename",
    description="d",
    category="test",
    replaces=node(
        "rename",
        title="Rename v2",
        description="For a test.",
        replaces=node("rename", title="Rename v1", description="For a test.")(rename_v1),
        upgrade=Upgrade(rewrite=rename_upgrade_1),
    )(rename_v2),
    upgrade=Upgrade(renamed_outputs={"result": "text"}),
)
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
            "r": NodeRecord(
                node="rename",
                version=version,
                inputs={"name": input_entry({"constant": "Ada Lovelace"})} if version == 1 else {},
            ),
            "e": NodeRecord(
                node="echo",
                inputs={
                    "text": input_entry(
                        {"sources": (Source("r", "result" if version < 3 else "text"),)}
                    )
                },
            ),
        },
        outputs={"result": Source("e", "result")},
    )


def _inputs(record: GraphRecord, node_id: str) -> dict:
    """Returns what each input of the node ``node_id`` gets, as it is in the record's JSON."""
    return record.model_dump(mode="json")["nodes"][node_id].get("inputs", {})


# --- an input or output the next version drops needs an upgrade ------------------


def _older_x(x: Annotated[Txt, _param("X")]) -> Annotated[Txt, Param(title="R")]:
    return x


def _newer_y(y: Annotated[Txt, _param("Y")]) -> Annotated[Txt, Param(title="R")]:
    return y


def test_an_input_gone_from_the_next_version_needs_an_upgrade():
    with pytest.raises(NodeDeclarationError) as raised:
        node(
            "two",
            title="Newer y",
            description="For a test.",
            replaces=node("two", title="Older x", description="For a test.")(_older_x),
        )(_newer_y)
    assert raised.value.problems[0].type == "upgrade_missing" and "two" in str(raised.value)


def test_an_upgrade_must_say_where_a_gone_input_went():
    with pytest.raises(NodeDeclarationError, match="upgrade|Upgrade"):
        node(
            "two",
            title="Newer y",
            description="For a test.",
            replaces=node("two", title="Older x", description="For a test.")(_older_x),
            upgrade=Upgrade(renamed_outputs={}),
        )(_newer_y)


def test_a_rename_or_a_rewrite_says_where_it_went():
    assert (
        node(
            "two",
            title="Newer y",
            description="For a test.",
            replaces=node("two", title="Older x", description="For a test.")(_older_x),
            upgrade=Upgrade(renamed_inputs={"x": "y"}),
        )(_newer_y).version
        == 2
    )
    assert (
        node(
            "two",
            title="Newer y",
            description="For a test.",
            replaces=node("two", title="Older x", description="For a test.")(_older_x),
            upgrade=Upgrade(rewrite=lambda v: {"y": v["x"]}),
        )(_newer_y).version
        == 2
    )


def test_a_version_keeping_its_fields_needs_no_upgrade():
    assert (
        node(
            "two",
            title="Older x",
            description="For a test.",
            replaces=node("two", title="Older x", description="For a test.")(_older_x),
        )(_older_x).version
        == 2
    )


# --- registry.upgrade ------------------------------------------------------------


def test_a_two_step_upgrade_rewrites_the_constants_and_sets_the_last_version():
    upgraded = _registry().upgrade(_graph(), "r")

    assert upgraded.nodes["r"].version == 3
    assert _inputs(upgraded, "r") == {
        "first": {"constant": "Ada"},
        "last": {"constant": "Lovelace"},
    }


def test_a_later_steps_rewrite_sees_the_constants_as_the_record_after_the_step_before_holds_them():
    """The second step's rewrite gets the constants the first step left, as the record holds them after it.

    The first rewrite returns a tuple and a date. The second sees them as the JSON a saved record keeps, a list
    and ISO text, the same as the record that ``to=2`` returns.
    """
    import datetime

    seen: list[dict] = []

    def first(before):
        return {"pair": (2, 1), "day": datetime.date(2026, 1, 2)}

    def second(before):
        seen.append(before)
        return before

    @node("steps", title="Steps v1", description="For a test.")
    def steps_v1(old: str) -> str:
        return old

    @node(
        "steps",
        title="Steps v2",
        description="For a test.",
        replaces=steps_v1,
        upgrade=Upgrade(rewrite=first),
    )
    def steps_v2(pair: list[int], day: str) -> str:
        return day

    @node(
        "steps",
        title="Steps",
        description="For a test.",
        replaces=steps_v2,
        upgrade=Upgrade(rewrite=second),
    )
    def steps_v3(pair: list[int], day: str) -> str:
        return day

    registry = NodeRegistry()
    registry.add(steps_v3)
    saved = GraphRecord.model_validate(
        {"nodes": {"s": {"node": "steps", "inputs": {"old": {"constant": "x"}}}}}
    )

    at_two = registry.upgrade(saved, "s", to=2)
    upgraded = registry.upgrade(saved, "s")

    assert seen == [{"pair": [2, 1], "day": "2026-01-02"}]
    assert {name: entry["constant"] for name, entry in _inputs(at_two, "s").items()} == seen[0]
    assert upgraded.nodes["s"].version == 3
    assert _inputs(upgraded, "s") == _inputs(at_two, "s")


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


def test_upgrade_reads_the_record_through_the_record_validator_and_refuses_two_nodes_of_one_id_or_an_id_that_is_no_name():
    two_of_one_id = '{"nodes": {"r": {"node": "rename"}, "r": {"node": "rename", "version": 2}}}'
    id_no_name = '{"nodes": {"a.b": {"node": "rename"}}}'

    with pytest.raises(InvalidCallError) as refused:
        _registry().upgrade(two_of_one_id, "r")
    assert [(p.type, p.ctx["errors"][0]["type"]) for p in refused.value.problems] == [
        ("invalid_record", "repeated_key")
    ]
    with pytest.raises(InvalidCallError) as refused:
        _registry().upgrade(id_no_name, "a.b")
    assert [(p.type, p.loc) for p in refused.value.problems] == [
        ("invalid_record", ("record", "nodes", "a.b", "[key]"))
    ]
    with pytest.raises(InvalidCallError) as refused:
        _registry().upgrade(_graph(), "nope")
    assert [(p.type, p.ctx["nodes"]) for p in refused.value.problems] == [
        ("node_not_in_graph", ["e", "r"])
    ]


def test_a_node_already_at_its_latest_version_returns_the_graph_unchanged():
    graph = _graph(version=3)
    assert _registry().upgrade(graph, "r") is graph


def test_an_edge_into_an_input_the_step_renames_moves_with_it():
    def carry_v1(
        name: Annotated[Txt, _param("Name")] = Txt(""),
    ) -> Annotated[Txt, Param(title="R")]:
        return name

    @node(
        "carry",
        title="Carry",
        description="d",
        category="test",
        replaces=node("carry", title="Carry v1", description="For a test.")(carry_v1),
        upgrade=Upgrade(renamed_inputs={"name": "full"}),
    )
    def carry(full: Annotated[Txt, _param("Full")] = Txt("")) -> Annotated[Txt, Param(title="R")]:
        return full

    registry = registry_of(echo, carry)
    graph = GraphRecord.model_validate(
        {
            "nodes": {
                "src": {"node": "echo"},
                "r": {
                    "node": "carry",
                    "inputs": {"name": {"sources": [{"node": "src", "output": "result"}]}},
                },
            }
        }
    )

    upgraded = registry.upgrade(graph, "r")
    assert _inputs(upgraded, "r") == {"full": {"sources": [{"node": "src", "output": "result"}]}}


def _versions_of_summed(kind: str, upgrade: Upgrade | None):
    """Declares two versions of the node ``summed``, as functions or as subgraphs.

    The first version takes ``a`` and ``b``. The second takes only ``a``.

    Returns:
        The newer version, and the nodes the subgraphs use.
    """

    @node("join", title="Join", description="For a test.")
    def join(a: int, b: int) -> int:
        return a + b

    @node("one", title="One", description="For a test.")
    def one(a: int) -> int:
        return a

    if kind == "function":
        older = node("summed", title="Summed v1", description="For a test.")(join.function)
        return node(
            "summed", title="Summed", description="For a test.", replaces=older, upgrade=upgrade
        )(one.function), []
    older = subgraph("summed", title="Summed v1", description="For a test.")(make_graph(join))
    return subgraph(
        "summed", title="Summed", description="For a test.", replaces=older, upgrade=upgrade
    )(make_graph(one)), [join, one]


_KINDS = pytest.mark.parametrize("kind", ["function", "subgraph"])


@_KINDS
def test_a_version_that_drops_an_input_needs_an_upgrade_for_a_function_and_a_subgraph_alike(
    kind: str,
) -> None:
    with pytest.raises(NodeDeclarationError) as raised:
        _versions_of_summed(kind, None)
    assert raised.value.problems[0].type == "upgrade_missing" and "summed" in str(raised.value)


@_KINDS
@pytest.mark.parametrize(
    "gone",
    [
        {"constant": 2},
        {"graph_input": "bee"},
        {"sources": [{"node": "t", "output": "result"}]},
    ],
    ids=["constant", "graph-input", "edge"],
)
def test_an_input_the_upgrade_removes_loses_its_constant_its_graph_input_or_its_edges(
    kind: str, gone: dict
) -> None:
    @node("three", title="Three", description="For a test.")
    def three() -> int:
        return 3

    summed, used = _versions_of_summed(
        kind, Upgrade(rewrite=lambda before: {k: v for k, v in before.items() if k != "b"})
    )
    registry = registry_of(three, summed, *used)
    graph = GraphRecord.model_validate(
        {
            "inputs": {"bee": {}} if "graph_input" in gone else {},
            "nodes": {
                "t": {"node": "three"},
                "s": {"node": "summed", "version": 1, "inputs": {"a": {"constant": 1}, "b": gone}},
            },
        }
    )
    assert Graph.compile(graph, registry).problems == []

    upgraded = registry.upgrade(graph, "s")
    assert _inputs(upgraded, "s") == {"a": {"constant": 1}}
    # a graph input that only the gone input took goes with it
    assert upgraded.inputs == {}
    assert Graph.compile(upgraded, registry).problems == []


def test_a_rewrite_sees_only_constants_and_a_graph_inputs_default_stays_the_graphs():
    def counted_v1(
        src: Annotated[Txt, _param("Source")] = Txt(""), n: int = 0
    ) -> Annotated[Txt, Param(title="R")]:
        return src

    @node(
        "counted",
        title="Counted",
        description="For a test.",
        replaces=node("counted", title="Counted v1", description="For a test.")(counted_v1),
        upgrade=Upgrade(rewrite=lambda before: {**before, "n": before.get("n", 0) + 1, "added": 1}),
    )
    def counted(
        src: Annotated[Txt, _param("Source")] = Txt(""), n: int = 0, added: int = 0
    ) -> Annotated[Txt, Param(title="R")]:
        return src

    registry = registry_of(counted)
    with_both = registry.upgrade(
        GraphRecord.model_validate(
            {
                "inputs": {"src": {"default": "x"}},
                "nodes": {
                    "c": {
                        "node": "counted",
                        "inputs": {
                            "n": {"constant": 2},
                            "src": {"graph_input": "src"},
                        },
                    }
                },
            }
        ),
        "c",
    )
    with_neither = registry.upgrade(
        GraphRecord.model_validate({"nodes": {"c": {"node": "counted"}}}), "c"
    )

    assert _inputs(with_both, "c") == {
        "n": {"constant": 3},
        "added": {"constant": 1},
        "src": {"graph_input": "src"},
    }
    assert with_both.model_dump(mode="json")["inputs"] == {"src": {"default": "x"}}
    assert _inputs(with_neither, "c") == {"n": {"constant": 1}, "added": {"constant": 1}}


def test_an_edge_reading_a_field_of_a_renamed_output_moves_with_it():
    registry = _registry()
    graph = GraphRecord.model_validate(
        {
            "nodes": {
                "r": {"node": "rename", "version": 2},
                "e": {
                    "node": "echo",
                    "inputs": {
                        "text": {"sources": [{"node": "r", "output": "result", "field": "text"}]}
                    },
                },
            }
        }
    )

    assert _inputs(registry.upgrade(graph, "r"), "e") == {
        "text": {"sources": [{"node": "r", "output": "text", "field": "text"}]}
    }


def test_a_downgrade_or_an_unknown_node_raises():
    registry = _registry()
    with pytest.raises(InvalidCallError, match="can't move from version 3 to version 1"):
        registry.upgrade(_graph(version=3), "r", to=1)
    with pytest.raises(InvalidCallError, match="nope"):
        registry.upgrade(_graph(), "nope")


def test_a_rewrite_that_raises_is_refused_at_the_node_naming_only_the_exceptions_class(caplog):
    """The rewrite is the node author's code. What it raises is ``cannot_move_to_version``, never a bare exception.

    The step that raised is in ``ctx``: here the second of two, from version 2 to 3. The exception's text goes to the
    ``conductor`` logger, and the problem names only its class.
    """

    def leaking(values):
        raise ValueError("the secret is 42")

    @node("leak", title="Leak v1", description="For a test.")
    def leak_v1(n: int) -> int:
        return n

    @node("leak", title="Leak v2", description="For a test.", replaces=leak_v1)
    def leak_v2(n: int) -> int:
        return n

    @node(
        "leak",
        title="Leak",
        description="For a test.",
        replaces=leak_v2,
        upgrade=Upgrade(rewrite=leaking),
    )
    def leak(n: int) -> int:
        return n

    saved = GraphRecord.model_validate(
        {"nodes": {"l": {"node": "leak", "inputs": {"n": {"constant": 1}}}}}
    )
    with (
        caplog.at_level(logging.ERROR, logger="conductor"),
        pytest.raises(InvalidCallError) as refused,
    ):
        registry_of(leak).upgrade(saved, "l")
    [problem] = refused.value.problems
    assert (problem.type, problem.loc, problem.ctx) == (
        "cannot_move_to_version_raised",
        ("l",),
        {"node": "leak", "version": 2, "target": 3, "error": "ValueError"},
    )
    assert "secret" not in problem.model_dump_json() and "the secret is 42" in caplog.text


def _bye_upgraded_by(rewrite) -> InvalidCallError:
    """The refusal of ``registry.upgrade`` for a node whose version 2 upgrades with ``rewrite``."""

    @node("bye", title="Bye v1", description="For a test.")
    def bye_v1(name: str) -> str:
        return name

    @node(
        "bye",
        title="Bye",
        description="For a test.",
        replaces=bye_v1,
        upgrade=Upgrade(rewrite=rewrite),
    )
    def bye(first: str) -> str:
        return first

    saved = GraphRecord.model_validate(
        {"nodes": {"b": {"node": "bye", "inputs": {"name": {"constant": "Ada Lovelace"}}}}}
    )
    with pytest.raises(InvalidCallError) as refused:
        registry_of(bye).upgrade(saved, "b")
    return refused.value


@pytest.mark.parametrize(
    ("returned", "reason", "error"),
    [
        (
            None,
            "a NoneType",
            {"type": "dict_type", "loc": [], "msg": "Input should be a valid dictionary"},
        ),
        (
            ["Ada"],
            "a list",
            {"type": "dict_type", "loc": [], "msg": "Input should be a valid dictionary"},
        ),
        (
            {1: "Ada"},
            "a dict with a key that isn't text",
            {"type": "string_type", "loc": [], "msg": "Input should be a valid string"},
        ),
    ],
)
def test_a_rewrite_that_returns_no_dict_of_input_names_is_refused_at_the_node_with_pydantics_error(
    returned, reason, error
):
    """A rewrite that forgets its ``return``, or returns a list, is refused at the node, never a bare exception.

    It is a kind of its own, with pydantic's error, as a hook's return is: ``ctx.error`` of the kind a ``NodeError``
    from the rewrite gives is only ever that error's code.
    """
    [problem] = _bye_upgraded_by(lambda values: returned).problems
    assert (problem.type, problem.loc, problem.ctx) == (
        "cannot_move_to_version_rewrite_returned_invalid",
        ("b",),
        {"node": "bye", "version": 1, "target": 2, "reason": reason, "errors": [error]},
    )
    assert problem.msg == (
        f"'bye' can't move from version 1 to version 2. The rewrite of its Upgrade returned {reason}."
        " Return a dict of input names to values."
    )


def test_a_rewrites_node_error_keeps_its_code_its_message_and_its_ctx():
    """A ``NodeError`` from a rewrite is kept as a hook's is, since its developer wrote it for people."""

    def first_name(values):
        raise NodeError("A full name can't be a first name.", code="full_name", ctx={"parts": 2})

    [problem] = _bye_upgraded_by(first_name).problems
    assert (problem.type, problem.loc, problem.ctx) == (
        "cannot_move_to_version_rewrite_failed",
        ("b",),
        {
            "node": "bye",
            "version": 1,
            "target": 2,
            "reason": "A full name can't be a first name.",
            "error": "full_name",
            "error_ctx": {"parts": 2},
        },
    )
    assert problem.msg == (
        "'bye' can't move from version 1 to version 2. The rewrite of its Upgrade failed: A full name can't be a"
        " first name."
    )


def test_compile_leaves_a_pinned_version_alone():
    registry = _registry()
    graph = Graph.compile(_graph(), registry)

    assert graph.describe().nodes["r"].version == 1
    assert graph.is_runnable, graph.problems
    assert graph.run().outputs == {"result": "Ada Lovelace"}


# --- what moves with a renamed field ---------------------------------------------


def renames_v1(name: Annotated[Txt, _param("Name")] = Txt("")) -> Annotated[Txt, Param(title="R")]:
    return name


@node(
    "renames",
    title="Renames",
    description="d",
    category="test",
    replaces=node("renames", title="Renames v1", description="For a test.")(renames_v1),
    upgrade=Upgrade(renamed_inputs={"name": "full"}, renamed_outputs={"result": "text"}),
)
def renames(full: Annotated[Txt, _param("Full")] = Txt("")) -> Named:
    """Version 2 renames the input ``name`` to ``full`` and the output ``result`` to ``text``."""
    return Named(text=full)


def test_a_renamed_input_takes_its_value_with_it():
    registry = registry_of(echo, renames)
    graph = GraphRecord.model_validate(
        {
            "nodes": {
                "r": {"node": "renames", "inputs": {"name": {"constant": "Ada"}}},
                "e": {
                    "node": "echo",
                    "inputs": {"text": {"sources": [{"node": "r", "output": "result"}]}},
                },
            }
        }
    )

    upgraded = registry.upgrade(graph, "r")

    assert _inputs(upgraded, "r") == {"full": {"constant": "Ada"}}
    assert _inputs(upgraded, "e") == {"text": {"sources": [{"node": "r", "output": "text"}]}}
    assert Graph.compile(upgraded, registry).problems == []


def test_an_edge_follows_its_output_when_a_later_step_reuses_the_outputs_old_name():
    """An edge follows its output through a rename, even when a later step reuses the output's old name.

    Step 1 renames ``result`` to ``first``. Step 2 adds a new output named ``result``, and renames that one to
    ``second``. An edge saved at version 1 read the output that version 3 calls ``first``, so it ends up there.
    """

    def reuse_outputs():
        return {}

    def reuse_v1() -> Annotated[Txt, Param(title="R")]:
        return Txt("")

    def reuse_v2() -> Mapping[str, Txt]:
        return {}

    @node(
        "reuse",
        title="Reuse",
        description="d",
        category="test",
        compute_outputs=reuse_outputs,
        replaces=node(
            "reuse",
            title="Reuse v2",
            description="For a test.",
            compute_outputs=reuse_outputs,
            replaces=node("reuse", title="Reuse v1", description="For a test.")(reuse_v1),
            upgrade=Upgrade(renamed_outputs={"result": "first"}),
        )(reuse_v2),
        upgrade=Upgrade(renamed_outputs={"result": "second"}),
    )
    def reuse() -> Mapping[str, Txt]:
        return {}

    registry = registry_of(echo, reuse)
    graph = GraphRecord.model_validate(
        {
            "nodes": {
                "u": {"node": "reuse"},
                "e": {
                    "node": "echo",
                    "inputs": {"text": {"sources": [{"node": "u", "output": "result"}]}},
                },
            }
        }
    )

    upgraded = registry.upgrade(graph, "u")

    assert _inputs(upgraded, "e") == {"text": {"sources": [{"node": "u", "output": "first"}]}}


def test_upgrading_a_subgraph_sets_only_its_version():
    """Upgrading a subgraph that has no ``Upgrade`` step changes only its version.

    The later version takes the same graph inputs, so the node keeps its inputs as they were saved.
    """

    def inner_graph(inner_id: str) -> GraphRecord:
        return GraphRecord(
            nodes={
                inner_id: NodeRecord(
                    node="echo", inputs={"text": input_entry({"graph_input": "text"})}
                )
            }
        )

    first = subgraph("inner_graph", title="Sub", description="d")(inner_graph("inner"))
    registry = registry_of(
        echo,
        subgraph("inner_graph", title="Sub", description="d", replaces=first)(
            inner_graph("renamed")
        ),
    )
    graph = GraphRecord.model_validate(
        {"nodes": {"f": {"node": "inner_graph", "inputs": {"text": {"constant": "x"}}}}}
    )

    upgraded = registry.upgrade(graph, "f")

    assert upgraded.nodes["f"].version == 2
    assert _inputs(upgraded, "f") == {"text": {"constant": "x"}}
    with pytest.raises(InvalidCallError, match="Its versions are 1, 2"):
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


@pytest.mark.parametrize(
    "upgrade", [Upgrade(renamed_inputs={}), Upgrade(rewrite=dict)], ids=["no-rename", "a-rewrite"]
)
def test_an_upgrade_that_drops_an_output_without_renaming_it_is_refused_where_it_is_declared(
    upgrade: Upgrade,
) -> None:
    """An upgrade that drops an output without renaming it is refused when the node is declared.

    A ``rewrite`` sees only the node's constants, never the edges out of it. So it can't say where a dropped output
    went.
    """
    with pytest.raises(
        NodeDeclarationError, match="the new version drops x, which the version it replaces has"
    ) as raised:
        node(
            "o",
            title="Newer out",
            description="For a test.",
            replaces=node("o", title="Older out", description="For a test.")(_older_out),
            upgrade=upgrade,
        )(_newer_out)
    assert raised.value.problems[0].type == "upgrade_missing"


def test_an_output_an_upgrade_renames_needs_nothing_more():
    assert (
        node(
            "o",
            title="Newer out",
            description="For a test.",
            replaces=node("o", title="Older out", description="For a test.")(_older_out),
            upgrade=Upgrade(renamed_outputs={"x": "y"}),
        )(_newer_out).version
        == 2
    )


@pytest.mark.parametrize(
    ("upgrade", "old", "new"),
    [
        (Upgrade(renamed_inputs={"z": "y"}, rewrite=dict), ["z"], []),
        (Upgrade(renamed_inputs={"x": "w"}, rewrite=dict), [], ["w"]),
        (Upgrade(renamed_outputs={"z": "result"}, rewrite=dict), ["z"], []),
        (Upgrade(renamed_outputs={"result": "w"}, rewrite=dict), [], ["w"]),
    ],
    ids=["old-input", "new-input", "old-output", "new-output"],
)
def test_an_upgrade_renaming_a_field_a_version_does_not_declare_is_refused_where_it_is_declared(
    upgrade: Upgrade, old: list[str], new: list[str]
):
    """A rename from a field the replaced version lacks, or to one the new version lacks, is refused at ``@node``.

    A name that is no field, such as a typo, moves nothing, so a graph upgraded by it would lose what that field got.
    The ``rewrite`` explains the dropped input, so the only fault is the rename.
    """
    with pytest.raises(NodeDeclarationError) as raised:
        node(
            "two",
            title="Newer y",
            description="For a test.",
            replaces=node("two", title="Older x", description="For a test.")(_older_x),
            upgrade=upgrade,
        )(_newer_y)

    (problem,) = raised.value.problems
    assert (problem.type, problem.loc, problem.ctx) == (
        "upgrade_renames_unknown",
        ("two",),
        {"old": old, "new": new},
    )
    assert str(raised.value).startswith("two (version 2): ")


def test_a_subgraph_upgrade_renaming_a_graph_input_it_does_not_have_is_refused():
    """A subgraph's renames are checked against the graph inputs and outputs of its two records."""
    inner = node("inner-x", title="Inner x", description="For a test.")(_older_x)
    record = make_graph(inner, inputs={"x": "inner-x.x"}, outputs={"r": "inner-x.result"})
    older = subgraph("sub", title="Sub", description="For a test.")(record)
    with pytest.raises(NodeDeclarationError) as raised:
        subgraph(
            "sub",
            title="Sub",
            description="For a test.",
            replaces=older,
            upgrade=Upgrade(renamed_inputs={"x": "y"}, renamed_outputs={"r": "s"}),
        )(record)

    (problem,) = raised.value.problems
    assert (problem.type, problem.ctx) == (
        "upgrade_renames_unknown",
        {"old": [], "new": ["y", "s"]},
    )


def test_a_rename_of_a_field_a_hook_may_give_is_not_checked():
    """A version with a hook may have fields it doesn't declare, so a rename of that kind of field passes ``@node``."""

    def outputs_by_name() -> dict[str, type]:
        return {"first": str}

    def older(**added: str) -> str:
        return ""

    def newer(kept: str) -> Mapping[str, str]:
        return {}

    replaced = node(
        "hooked",
        title="Hooked v1",
        description="For a test.",
        compute_added_inputs=lambda: {"extra": str},
    )(older)
    assert (
        node(
            "hooked",
            title="Hooked",
            description="For a test.",
            compute_outputs=outputs_by_name,
            replaces=replaced,
            upgrade=Upgrade(
                renamed_inputs={"extra": "kept"}, renamed_outputs={"result": "first"}, rewrite=dict
            ),
        )(newer).version
        == 2
    )


def test_a_rename_into_inputs_the_new_versions_hook_adds_is_not_checked():
    """A new version that adds inputs with a hook may get the input a rename ends at, so ``@node`` lets it pass."""

    def newer(**added: str) -> Annotated[Txt, Param(title="R")]:
        return ""

    assert (
        node(
            "two",
            title="Newer added",
            description="For a test.",
            compute_added_inputs=lambda: {"extra": str},
            replaces=node("two", title="Older x", description="For a test.")(_older_x),
            upgrade=Upgrade(renamed_inputs={"x": "extra"}),
        )(newer).version
        == 2
    )


def test_a_rename_from_outputs_the_old_versions_hook_computes_is_not_checked():
    """An old version whose hook computes its outputs may have had the output a rename starts at."""

    def older(x: Annotated[Txt, _param("X")]) -> Mapping[str, str]:
        return {}

    assert (
        node(
            "two",
            title="Newer x",
            description="For a test.",
            replaces=node(
                "two",
                title="Older computed",
                description="For a test.",
                compute_outputs=lambda: {"first": str},
            )(older),
            upgrade=Upgrade(renamed_outputs={"first": "result"}),
        )(_older_x).version
        == 2
    )


def test_a_rewrite_returning_a_set_is_refused_as_set_has_no_order_at_the_node():
    def tagged_v1(tag: str) -> str:
        return tag

    @node(
        "tagged",
        title="Tagged",
        description="For a test.",
        replaces=node("tagged", title="Tagged v1", description="For a test.")(tagged_v1),
        upgrade=Upgrade(rewrite=lambda before: {"tags": set(before.values())}),
    )
    def tagged(tags: list[str]) -> str:
        return ",".join(tags)

    with pytest.raises(InvalidCallError) as refused:
        registry_of(tagged).upgrade(
            GraphRecord.model_validate(
                {"nodes": {"t": {"node": "tagged", "inputs": {"tag": {"constant": "a"}}}}}
            ),
            "t",
        )
    (problem,) = refused.value.problems
    assert (problem.type, problem.loc, [e["type"] for e in problem.ctx["errors"]]) == (
        "invalid_value",
        ("t", "tags"),
        ["set_has_no_order"],
    )


def test_a_rewrite_is_written_as_json_and_one_with_no_json_form_is_invalid_value_at_the_node():
    import datetime

    import polars as pl

    @node("when", title="When v1", description="For a test.")
    def when_v1(day: str) -> str:
        return day

    saved = GraphRecord.model_validate(
        {"nodes": {"w": {"node": "when", "inputs": {"day": {"constant": "2026-01-02"}}}}}
    )
    for returned, written in (
        (datetime.date(2026, 1, 2), "2026-01-02"),
        (pl.DataFrame({"a": [1]}), None),
    ):

        @node(
            "when",
            title="When",
            description="For a test.",
            replaces=when_v1,
            upgrade=Upgrade(rewrite=lambda before, r=returned: {"day": r}),
        )
        def when(day: datetime.date) -> str:
            return day.isoformat()

        registry = NodeRegistry()
        registry.add(when)
        if written is None:
            with pytest.raises(InvalidCallError) as refused:
                registry.upgrade(saved, "w")
            assert [(p.type, p.loc) for p in refused.value.problems] == [
                ("invalid_value", ("w", "day"))
            ]
        else:
            upgraded = registry.upgrade(saved, "w")
            assert _inputs(upgraded, "w") == {"day": {"constant": written}}
            assert GraphRecord.model_validate_json(upgraded.model_dump_json()) == upgraded


def test_a_subgraphs_input_rename_moves_only_the_edge_into_that_exact_input():
    """Renaming a subgraph's input moves only the edges into that exact input.

    The subgraph's inputs are ``a``, ``ab`` and ``cb``. Renaming ``a`` leaves the edges into ``ab`` and ``cb`` alone.
    """

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
        return GraphRecord.model_validate(
            {
                "inputs": {"ab": {}, "cb": {}, name: {}},
                "nodes": {
                    "a": {"node": "takes-b", "inputs": {"b": {"graph_input": "ab"}}},
                    "c": {"node": "takes-b", "inputs": {"b": {"graph_input": "cb"}}},
                    "x": {"node": last, "inputs": {name: {"graph_input": name}}},
                },
            }
        )

    v1 = subgraph("sub", title="Sub", description="For a test.")(inner("takes-a", "a"))
    v2 = subgraph(
        "sub",
        title="Sub",
        description="For a test.",
        replaces=v1,
        upgrade=Upgrade(renamed_inputs={"a": "z"}),
    )(inner("takes-z", "z"))
    reg = registry_of(takes_b, takes_a, takes_z, word, v2)
    word_result = {"sources": [{"node": "w", "output": "result"}]}
    stored = GraphRecord.model_validate(
        {
            "nodes": {
                "w": {"node": "word"},
                "s": {
                    "node": "sub",
                    "version": 1,
                    "inputs": {"ab": word_result, "cb": word_result, "a": word_result},
                },
            }
        }
    )
    assert Graph.compile(stored, reg).problems == []

    moved = reg.upgrade(stored, "s")

    assert _inputs(moved, "s") == {"ab": word_result, "cb": word_result, "z": word_result}
    assert Graph.compile(moved, reg).problems == []


def test_a_rename_keeps_a_constant_a_graph_input_and_edges_as_they_are():
    def carry_v1(a: str, b: str, c: str) -> str:
        return a + b + c

    @node(
        "carry3",
        title="Carry",
        description="For a test.",
        replaces=node("carry3", title="Carry v1", description="For a test.")(carry_v1),
        upgrade=Upgrade(renamed_inputs={"a": "x", "b": "y", "c": "z"}),
    )
    def carry(x: str, y: str, z: str) -> str:
        return x + y + z

    registry = registry_of(echo, carry)
    graph = GraphRecord.model_validate(
        {
            "inputs": {"bee": {"default": "2"}},
            "nodes": {
                "src": {"node": "echo"},
                "k": {
                    "node": "carry3",
                    "inputs": {
                        "a": {"constant": "1"},
                        "b": {"graph_input": "bee"},
                        "c": {"sources": [{"node": "src", "output": "result"}]},
                    },
                },
            },
            "outputs": {"result": {"node": "k", "output": "result"}},
        }
    )

    upgraded = registry.upgrade(graph, "k")

    assert _inputs(upgraded, "k") == {
        "x": {"constant": "1"},
        "y": {"graph_input": "bee"},
        "z": {"sources": [{"node": "src", "output": "result"}]},
    }
    assert Graph.compile(upgraded, registry).problems == []


@pytest.mark.parametrize(
    ("declared", "entry"),
    [
        ({}, {"sources": [{"node": "src", "output": "result"}]}),
        ({"text": {"default": "a"}}, {"graph_input": "text"}),
    ],
    ids=["edges", "graph_input"],
)
def test_a_rewrite_giving_a_value_to_an_input_with_edges_or_a_graph_input_is_refused(
    declared: dict, entry: dict
):
    """A rewrite that gives a value to an input with edges into it, or a graph input, is refused.

    An input gets exactly one of a constant, a graph input or edges. A value on such an input would give it two.
    So the record's validator refuses what the step wrote. A graph input's default is the whole graph's, so a
    rewrite never turns it into a node's constant.
    """

    def fed_v1(text: str) -> str:
        return text

    @node(
        "fed",
        title="Fed",
        description="For a test.",
        replaces=node("fed", title="Fed v1", description="For a test.")(fed_v1),
        upgrade=Upgrade(rewrite=lambda before: {**before, "text": "x"}),
    )
    def fed(text: str) -> str:
        return text

    registry = registry_of(echo, fed)
    graph = GraphRecord.model_validate(
        {
            "inputs": declared,
            "nodes": {"src": {"node": "echo"}, "f": {"node": "fed", "inputs": {"text": entry}}},
        }
    )

    with pytest.raises(InvalidCallError) as refused:
        registry.upgrade(graph, "f")
    [problem] = refused.value.problems
    assert (
        problem.type,
        problem.loc,
        problem.ctx["errors"][0]["type"],
        problem.ctx["written_by"],
    ) == (
        "invalid_record",
        ("record", "nodes", "f", "inputs", "text"),
        "one_of_constant_graph_input_sources",
        "the Upgrade of 'fed' to version 2",
    )
    assert problem.msg.endswith("Fix the Upgrade of 'fed' to version 2.")
