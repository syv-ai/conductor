"""The standard node library, node by node, through ``compile`` + ``run_sync``.

Each node is put in a graph and run, so registration, the declared
interface and the behaviour of ``run`` are checked together. Results come
back as the library's own dtypes (``str``, ``float``, ``bool``, ``Json``);
a series output compares as a list. ``decision`` is called directly: its
``value`` is ``Any`` and nothing here records what arrives on it.
"""

from __future__ import annotations

import conductor_nodes
import pytest
from conductor import SKIPPED, CompiledGraph, From, Graph, GraphNode, NodeRegistry, Static
from conductor_nodes.decision import decision
from conductor_nodes.types import Json
from helpers import stream_of


@pytest.fixture
def full_registry() -> NodeRegistry:
    reg = conductor_nodes.registry()
    return reg


def _ending(reg: NodeRegistry, nodes):
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), reg)
    return compiled.run_sync()


def _run(reg: NodeRegistry, nodes):
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), reg)
    return compiled.results(compiled.run_sync())


@pytest.mark.parametrize("type_id, inputs, expected", [
    ("text-uppercase", {"text": "hello"}, "HELLO"),
    ("text-lowercase", {"text": "HELLO"}, "hello"),
    ("text-trim", {"text": "  hi  "}, "hi"),
    ("text-length", {"text": "hello"}, 5),
    ("text-concat", {"a": "foo", "b": "bar", "separator": "-"}, "foo-bar"),
    ("text-replace", {"text": "hello world", "needle": "world", "replacement": "there"}, "hello there"),
    ("text-contains", {"text": "Hello", "needle": "hello", "case_sensitive": False}, True),
    ("text-contains", {"text": "Hello", "needle": "hello", "case_sensitive": True}, False),
    ("text-join", {"parts": ["a", "b", "c"], "separator": "-"}, "a-b-c"),
    ("text-reverse", {"text": "hello"}, "olleh"),
    ("math-add", {"a": 2, "b": 3}, 5),
    ("math-subtract", {"a": 5, "b": 3}, 2),
    ("math-multiply", {"a": 4, "b": 3}, 12),
    ("math-divide", {"a": 10, "b": 4}, 2.5),
    ("math-modulo", {"a": 10, "b": 3}, 1),
    ("math-round", {"value": 3.7}, 4),
    ("math-round", {"value": 3.14159, "decimals": 2}, 3.14),
    ("math-min", {"values": [3, 1, 2]}, 1),
    ("math-max", {"values": [3, 1, 2]}, 3),
    ("math-abs", {"value": -7}, 7),
    ("logic-not", {"value": True}, False),
    ("json-parse", {"text": '{"a": 1, "b": [2, 3]}'}, Json({"a": 1, "b": [2, 3]})),
    ("json-stringify", {"value": {"b": 2, "a": 1}, "sort_keys": True}, '{"a": 1, "b": 2}'),
    ("json-get", {"value": {"user": {"name": "Ada"}}, "path": "user.name"}, Json("Ada")),
    ("json-get", {"value": {"items": [{"id": "a"}, {"id": "b"}]}, "path": "items.1.id"}, Json("b")),
    ("json-get", {"value": {"a": 1}, "path": "does.not.exist"}, Json(None)),
    ("regex-match", {"text": "hello 123 world", "pattern": r"\d+"}, True),
    ("regex-match", {"text": "nothing numeric", "pattern": r"\d+"}, False),
    ("regex-match", {"text": "Hello", "pattern": r"hello", "ignore_case": True}, True),
    ("regex-replace", {"text": "a1b2c3", "pattern": r"\d", "replacement": "-"}, "a-b-c-"),
])
def test_a_node_run_on_static_inputs_gives_its_result(full_registry, type_id, inputs, expected):
    bindings = {name: Static(value) for name, value in inputs.items()}
    r = _run(full_registry, [GraphNode(id="n", type=type_id, version=1, bindings=bindings)])
    assert r["n"]["result"] == expected


class TestPackageSurface:
    def test_register_all_registers_every_category(self, full_registry):
        ids = {c.id for c in full_registry.nodes}
        # Spot-check one node per category
        assert "text-uppercase" in ids
        assert "math-add" in ids
        assert "logic-if-empty" in ids
        assert "json-parse" in ids
        assert "regex-match" in ids
        assert "decision" in ids

    def test_register_all_respects_categories_filter(self):
        reg = NodeRegistry()
        reg = conductor_nodes.registry(categories=["text", "math"])
        ids = {c.id for c in reg.nodes}
        assert "text-uppercase" in ids
        assert "math-add" in ids
        assert "json-parse" not in ids
        assert "logic-if-empty" not in ids

    def test_register_all_rejects_unknown_category(self):
        with pytest.raises(KeyError, match="Unknown category"):
            conductor_nodes.registry(categories=["doesnt-exist"])

    def test_each_module_lists_its_own_nodes(self):
        ids = {c.id for c in conductor_nodes.text.NODES}
        assert "text-uppercase" in ids
        assert "math-add" not in ids   # only text's own nodes
        assert {c.id for c in NodeRegistry(nodes=conductor_nodes.text.NODES).nodes} == ids


class TestText:
    def test_split(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="text-split", version=1, bindings={"text": Static("a,b,c"), "separator": Static(",")})],
        )
        assert r["n"]["result"]["values"] == ["a", "b", "c"]


class TestMath:
    def test_divide_by_zero_raises(self, full_registry):
        ending = _ending(
            full_registry,
            [GraphNode(id="n", type="math-divide", version=1, bindings={"a": Static(1), "b": Static(0)})],
        )
        assert ending.status == "failed"


class TestLogic:
    """The two ``if`` nodes route text to one named output and ``SKIPPED`` on the other."""

    def test_if_empty_routes_to_empty_branch(self, full_registry):
        r = _run(
            full_registry,
            [
                GraphNode(id="cond", type="logic-if-empty", version=1, bindings={"text": Static("   ")}),
                GraphNode(id="down", type="text-uppercase", version=1, bindings={"text": From('cond.empty')}),
            ],
        )
        # Empty branch delivered "   " to the downstream node
        assert r["down"]["result"] == "   "

    def test_if_empty_routes_to_non_empty_branch(self, full_registry):
        # The "empty" branch consumer should be skipped.
        r = _run(
            full_registry,
            [
                GraphNode(id="cond", type="logic-if-empty", version=1, bindings={"text": Static("hi")}),
                GraphNode(id="up", type="text-uppercase", version=1, bindings={"text": From('cond.not_empty')}),
                GraphNode(id="other", type="text-uppercase", version=1, bindings={"text": From('cond.empty')}),
            ],
        )
        assert r["up"]["result"] == "HI"
        assert "other" not in r   # skipped

    def test_if_equals_true(self, full_registry):
        r = _run(
            full_registry,
            [
                GraphNode(id="cond", type="logic-if-equals", version=1, bindings={"a": Static("foo"), "b": Static("foo")}),
                GraphNode(id="eq", type="text-uppercase", version=1, bindings={"text": From('cond.equal')}),
            ],
        )
        assert r["eq"]["result"] == "FOO"

    def test_if_equals_case_insensitive(self, full_registry):
        r = _run(
            full_registry,
            [
                GraphNode(id="cond", type="logic-if-equals", version=1,
                          bindings={"a": Static("Foo"), "b": Static("FOO"), "case_sensitive": Static(False)}),
                GraphNode(id="eq", type="text-uppercase", version=1, bindings={"text": From('cond.equal')}),
            ],
        )
        assert r["eq"]["result"] == "FOO"


class TestDecision:
    """The gate's body: ``value`` lands on the branch ``when`` picks, ``SKIPPED`` on the other."""

    def test_true_routes_to_if_true(self):
        branches = decision(value="x", when=True)
        assert branches.if_true == "x"
        assert branches.if_false is SKIPPED

    def test_false_routes_to_if_false(self):
        branches = decision(value="x", when=False)
        assert branches.if_true is SKIPPED
        assert branches.if_false == "x"


class TestJSON:
    def test_stringify_indented(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="json-stringify", version=1, bindings={"value": Static({"x": 1}), "indent": Static(2)})],
        )
        assert "\n" in r["n"]["result"]


class TestRegex:
    def test_extract_findall_when_no_groups(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="regex-extract", version=1,
                       bindings={"text": Static("a1 b22 c333"), "pattern": Static(r"\d+")})],
        )
        assert r["n"]["result"]["values"] == ["1", "22", "333"]

    def test_extract_uses_first_group(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="regex-extract", version=1,
                       bindings={"text": Static("name=Ada, age=36"), "pattern": Static(r"name=(\w+)")})],
        )
        assert r["n"]["result"]["values"] == ["Ada"]


class TestIntegration:
    """Several categories chained through edges."""

    def test_pipeline_split_trim_join_concat(self, full_registry):
        r = _run(
            full_registry,
            [
                GraphNode(id="src", type="text-split", version=1,
                          bindings={"text": Static(" a ,  b , c "), "separator": Static(",")}),
                # Reuse the split result, upper-cased after a join
                GraphNode(id="joined", type="text-join", version=1, bindings={"separator": Static("|"), "parts": From('src.result')}),
                GraphNode(id="upper", type="text-uppercase", version=1, bindings={"text": From('joined.result')}),
            ],
        )
        # split produces [" a ", "  b ", " c "], join preserves whitespace,
        # upper just uppercases — no trim in the pipeline on purpose so the
        # test documents the raw composition.
        assert r["upper"]["result"] == " A |  B | C "

    def test_math_pipeline_via_edges(self, full_registry):
        r = _run(
            full_registry,
            [
                GraphNode(id="a", type="math-add", version=1, bindings={"a": Static(2), "b": Static(3)}),     # 5
                GraphNode(id="b", type="math-multiply", version=1, bindings={"b": Static(4), "a": From('a.result')}),          # 5 * 4 = 20
                GraphNode(id="c", type="math-round", version=1, bindings={"decimals": Static(0), "value": From('b.result')}),      # 20
            ],
        )
        assert r["c"]["result"] == 20


# --- the regex nodes cannot freeze a run, and categories filter by category ---------


def test_a_catastrophic_pattern_answers_at_once():
    """``(a+)+$`` over 27 a's and a b held the GIL for seconds under ``re``;
    the ``regex`` engine does not backtrack on it."""
    import asyncio
    import time


    graph = Graph(nodes=[GraphNode(id="m", type="regex-match", version=1, bindings={
        "text": Static("a" * 27 + "b"), "pattern": Static("(a+)+$"),
    })])
    compiled = CompiledGraph.from_graph(graph, conductor_nodes.registry())

    async def events():
        return [event async for event in stream_of(compiled)]

    started = time.monotonic()
    seen = asyncio.run(events())
    assert time.monotonic() - started < conductor_nodes.regex_ops.TIMEOUT
    assert "node_start" in [e.type for e in seen]
    assert not compiled.results(seen[-1].state)["m"]["result"]


def test_a_pattern_that_times_out_fails_its_node_with_a_sentence_for_people(monkeypatch):
    import regex

    def too_slow(*args, **kwargs):
        raise TimeoutError("regex timed out")

    monkeypatch.setattr(regex, "search", too_slow)
    graph = Graph(nodes=[GraphNode(id="m", type="regex-match", version=1, bindings={"text": Static("x"), "pattern": Static("x")})])
    ending = CompiledGraph.from_graph(graph, conductor_nodes.registry()).run_sync()

    assert ending.status == "failed"
    assert (ending.cause.type, ending.cause.msg) == ("pattern_timeout", "The pattern took too long.")


def test_a_grouped_extract_that_times_out_fails_with_the_same_code(monkeypatch):
    """With groups, Extract walks the matches one by one, and the timeout
    fires during the walk rather than when the search starts."""
    monkeypatch.setattr(conductor_nodes.regex_ops, "TIMEOUT", 0.05)
    graph = Graph(nodes=[GraphNode(id="x", type="regex-extract", version=1, bindings={
        "text": Static("a" * 40), "pattern": Static("((?:(?:a|aa)+)+b)"),
    })])
    ending = CompiledGraph.from_graph(graph, conductor_nodes.registry()).run_sync()

    assert ending.status == "failed"
    assert ending.cause.type == "pattern_timeout"


def test_categories_filter_on_each_nodes_own_category():
    logic = conductor_nodes.registry(categories=["logic"])
    control = conductor_nodes.registry(categories=["control"])

    assert {cls.category for cls in logic.nodes} == {"logic"}
    assert {cls.id for cls in control.nodes} == {"decision", "logic-if-empty", "logic-if-equals"}


def test_a_category_given_as_a_string_is_refused():
    with pytest.raises(TypeError, match="list of category names"):
        conductor_nodes.registry(categories="text")


def test_there_is_no_category_table():
    assert not hasattr(conductor_nodes, "CATEGORIES")
