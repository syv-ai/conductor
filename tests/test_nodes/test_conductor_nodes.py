"""The standard node library, node by node, through ``compile`` + ``run_sync``.

Each node is placed in a graph and run, so registration, the declared
interface and the behaviour of ``run`` are checked together. Results come
back as the library's own dtypes (``Text``, ``Number``, ``Flag``, ``Json``);
a series output compares as a list. ``decision`` is called directly: its
``value`` is ``Any`` and nothing here records what arrives on it.
"""

from __future__ import annotations

import conductor_nodes
import pytest
from conductor import CompiledGraph, GraphNode, NodeRegistry, run_sync
from conductor._sentinel import SKIPPED
from conductor.graph.binding import From, Static
from conductor.graph.model import Graph
from conductor_nodes.decision import Decision
from conductor_nodes.types import Flag, Json, Text


@pytest.fixture
def full_registry() -> NodeRegistry:
    reg = NodeRegistry()
    conductor_nodes.register_all(reg)
    return reg


def _ending(reg: NodeRegistry, nodes):
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), reg)
    return run_sync(compiled)


def _run(reg: NodeRegistry, nodes):
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), reg)
    return run_sync(compiled).state.results(compiled)


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
        conductor_nodes.register_all(reg, categories=["text", "math"])
        ids = {c.id for c in reg.nodes}
        assert "text-uppercase" in ids
        assert "math-add" in ids
        assert "json-parse" not in ids
        assert "logic-if-empty" not in ids

    def test_register_all_rejects_unknown_category(self):
        reg = NodeRegistry()
        with pytest.raises(KeyError, match="Unknown category"):
            conductor_nodes.register_all(reg, categories=["doesnt-exist"])

    def test_individual_modules_expose_register(self):
        reg = NodeRegistry()
        conductor_nodes.text.register(reg)
        ids = {c.id for c in reg.nodes}
        assert "text-uppercase" in ids
        assert "math-add" not in ids   # only text registered


class TestText:
    def test_uppercase(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="text-uppercase", version=1, bindings={"text": Static("hello")})],
        )
        assert r["n"]["result"] == "HELLO"

    def test_lowercase(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="text-lowercase", version=1, bindings={"text": Static("HELLO")})],
        )
        assert r["n"]["result"] == "hello"

    def test_trim(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="text-trim", version=1, bindings={"text": Static("  hi  ")})],
        )
        assert r["n"]["result"] == "hi"

    def test_length(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="text-length", version=1, bindings={"text": Static("hello")})],
        )
        assert r["n"]["result"] == 5

    def test_concat_with_separator(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="text-concat", version=1, bindings={"a": Static("foo"), "b": Static("bar"), "separator": Static("-")})],
        )
        assert r["n"]["result"] == "foo-bar"

    def test_replace(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="text-replace", version=1,
                       bindings={"text": Static("hello world"), "needle": Static("world"), "replacement": Static("there")})],
        )
        assert r["n"]["result"] == "hello there"

    def test_contains_case_insensitive(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="text-contains", version=1,
                       bindings={"text": Static("Hello"), "needle": Static("hello"), "case_sensitive": Static(False)})],
        )
        assert r["n"]["result"] == Flag(True)

    def test_contains_case_sensitive(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="text-contains", version=1,
                       bindings={"text": Static("Hello"), "needle": Static("hello"), "case_sensitive": Static(True)})],
        )
        assert r["n"]["result"] == Flag(False)

    def test_split(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="text-split", version=1, bindings={"text": Static("a,b,c"), "separator": Static(",")})],
        )
        assert list(r["n"]["result"]) == ["a", "b", "c"]

    def test_join(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="text-join", version=1, bindings={"parts": Static(["a", "b", "c"]), "separator": Static("-")})],
        )
        assert r["n"]["result"] == "a-b-c"

    def test_reverse(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="text-reverse", version=1, bindings={"text": Static("hello")})],
        )
        assert r["n"]["result"] == "olleh"


class TestMath:
    def test_add(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="math-add", version=1, bindings={"a": Static(2), "b": Static(3)})],
        )
        assert r["n"]["result"] == 5

    def test_subtract(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="math-subtract", version=1, bindings={"a": Static(5), "b": Static(3)})],
        )
        assert r["n"]["result"] == 2

    def test_multiply(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="math-multiply", version=1, bindings={"a": Static(4), "b": Static(3)})],
        )
        assert r["n"]["result"] == 12

    def test_divide(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="math-divide", version=1, bindings={"a": Static(10), "b": Static(4)})],
        )
        assert r["n"]["result"] == 2.5

    def test_divide_by_zero_raises(self, full_registry):
        ending = _ending(
            full_registry,
            [GraphNode(id="n", type="math-divide", version=1, bindings={"a": Static(1), "b": Static(0)})],
        )
        assert ending.type == "graph_error"

    def test_modulo(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="math-modulo", version=1, bindings={"a": Static(10), "b": Static(3)})],
        )
        assert r["n"]["result"] == 1

    def test_round_default(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="math-round", version=1, bindings={"value": Static(3.7)})],
        )
        assert r["n"]["result"] == 4

    def test_round_to_decimals(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="math-round", version=1, bindings={"value": Static(3.14159), "decimals": Static(2)})],
        )
        assert r["n"]["result"] == 3.14

    def test_min(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="math-min", version=1, bindings={"values": Static([3, 1, 2])})],
        )
        assert r["n"]["result"] == 1

    def test_max(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="math-max", version=1, bindings={"values": Static([3, 1, 2])})],
        )
        assert r["n"]["result"] == 3

    def test_abs(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="math-abs", version=1, bindings={"value": Static(-7)})],
        )
        assert r["n"]["result"] == 7


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

    def test_not(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="logic-not", version=1, bindings={"value": Static(True)})],
        )
        assert r["n"]["result"] == Flag(False)


class TestDecision:
    """The gate's body: ``value`` lands on the branch ``when`` picks, ``SKIPPED`` on the other."""

    def test_true_routes_to_if_true(self):
        branches = Decision().run(value=Text("x"), when=Flag(True))
        assert branches.if_true == Text("x")
        assert branches.if_false is SKIPPED

    def test_false_routes_to_if_false(self):
        branches = Decision().run(value=Text("x"), when=Flag(False))
        assert branches.if_true is SKIPPED
        assert branches.if_false == Text("x")


class TestJSON:
    def test_parse(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="json-parse", version=1, bindings={"text": Static('{"a": 1, "b": [2, 3]}')})],
        )
        assert r["n"]["result"] == Json({"a": 1, "b": [2, 3]})

    def test_stringify(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="json-stringify", version=1,
                       bindings={"value": Static({"b": 2, "a": 1}), "sort_keys": Static(True)})],
        )
        assert r["n"]["result"] == '{"a": 1, "b": 2}'

    def test_stringify_indented(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="json-stringify", version=1, bindings={"value": Static({"x": 1}), "indent": Static(2)})],
        )
        assert "\n" in r["n"]["result"]

    def test_get_simple_key(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="json-get", version=1,
                       bindings={"value": Static({"user": {"name": "Ada"}}), "path": Static("user.name")})],
        )
        assert r["n"]["result"] == Json("Ada")

    def test_get_list_index(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="json-get", version=1,
                       bindings={"value": Static({"items": [{"id": "a"}, {"id": "b"}]}), "path": Static("items.1.id")})],
        )
        assert r["n"]["result"] == Json("b")

    def test_get_missing_path_returns_none(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="json-get", version=1,
                       bindings={"value": Static({"a": 1}), "path": Static("does.not.exist")})],
        )
        assert r["n"]["result"] == Json(None)


class TestRegex:
    def test_match_true(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="regex-match", version=1,
                       bindings={"text": Static("hello 123 world"), "pattern": Static(r"\d+")})],
        )
        assert r["n"]["result"] == Flag(True)

    def test_match_false(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="regex-match", version=1,
                       bindings={"text": Static("nothing numeric"), "pattern": Static(r"\d+")})],
        )
        assert r["n"]["result"] == Flag(False)

    def test_match_ignore_case(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="regex-match", version=1,
                       bindings={"text": Static("Hello"), "pattern": Static(r"hello"), "ignore_case": Static(True)})],
        )
        assert r["n"]["result"] == Flag(True)

    def test_replace(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="regex-replace", version=1,
                       bindings={"text": Static("a1b2c3"), "pattern": Static(r"\d"), "replacement": Static("-")})],
        )
        assert r["n"]["result"] == "a-b-c-"

    def test_extract_findall_when_no_groups(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="regex-extract", version=1,
                       bindings={"text": Static("a1 b22 c333"), "pattern": Static(r"\d+")})],
        )
        assert list(r["n"]["result"]) == ["1", "22", "333"]

    def test_extract_uses_first_group(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode(id="n", type="regex-extract", version=1,
                       bindings={"text": Static("name=Ada, age=36"), "pattern": Static(r"name=(\w+)")})],
        )
        assert list(r["n"]["result"]) == ["Ada"]


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

    from conductor import execute

    graph = Graph(nodes=[GraphNode(id="m", type="regex-match", version=1, bindings={
        "text": Static("a" * 27 + "b"), "pattern": Static("(a+)+$"),
    })])
    compiled = CompiledGraph.from_graph(graph, conductor_nodes.registry())

    async def events():
        return [event async for event in execute(compiled)]

    started = time.monotonic()
    seen = asyncio.run(events())
    assert time.monotonic() - started < conductor_nodes.regex_ops.Match.timeout
    assert "node_start" in [e.type for e in seen]
    assert seen[-1].state.results(compiled)["m"]["result"] == Flag(False)


def test_a_pattern_that_times_out_fails_its_node_with_a_sentence_for_people(monkeypatch):
    import regex

    def too_slow(*args, **kwargs):
        raise TimeoutError("regex timed out")

    monkeypatch.setattr(regex, "search", too_slow)
    graph = Graph(nodes=[GraphNode(id="m", type="regex-match", version=1, bindings={"text": Static("x"), "pattern": Static("x")})])
    ending = run_sync(CompiledGraph.from_graph(graph, conductor_nodes.registry()))

    assert ending.type == "graph_error"
    assert (ending.cause.code, ending.error) == ("pattern_timeout", "The pattern took too long.")


def test_a_grouped_extract_that_times_out_fails_with_the_same_code(monkeypatch):
    """With groups, Extract walks the matches one by one, and the timeout
    fires during the walk rather than when the search starts."""
    monkeypatch.setattr(conductor_nodes.regex_ops.Extract, "timeout", 0.05)
    graph = Graph(nodes=[GraphNode(id="x", type="regex-extract", version=1, bindings={
        "text": Static("a" * 40), "pattern": Static("((?:(?:a|aa)+)+b)"),
    })])
    ending = run_sync(CompiledGraph.from_graph(graph, conductor_nodes.registry()))

    assert ending.type == "graph_error"
    assert ending.cause.code == "pattern_timeout"


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
