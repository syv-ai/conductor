"""The standard node library, node by node, through ``compile`` + ``execute_sync``.

Each node is placed in a graph and run, so registration, the declared
interface and the behaviour of ``run`` are checked together. Results come
back as the library's own dtypes (``Text``, ``Number``, ``Flag``, ``Json``);
a series output compares as a list. ``decision`` is called directly: its
``value`` is ``Any`` and nothing here records what arrives on it.
"""

from __future__ import annotations

import conductor_nodes
import pytest
from conductor import GraphNode, NodeRegistry, compile
from conductor._sentinel import SKIPPED
from conductor.errors import FlowExecutionError
from conductor.execution.engine import execute_sync
from conductor.graph.binding import Sources, Static
from conductor.graph.model import Flow
from conductor.ref import Ref
from conductor_nodes.decision import Decision
from conductor_nodes.types import Flag, Json, Text


@pytest.fixture
def full_registry() -> NodeRegistry:
    reg = NodeRegistry()
    conductor_nodes.register_all(reg)
    return reg


def _run(reg: NodeRegistry, nodes):
    compiled = compile(Flow(nodes=nodes), reg)
    return execute_sync(compiled)


class TestPackageSurface:
    def test_register_all_registers_every_category(self, full_registry):
        ids = {c.id for c in full_registry.definitions()}
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
        ids = {c.id for c in reg.definitions()}
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
        ids = {c.id for c in reg.definitions()}
        assert "text-uppercase" in ids
        assert "math-add" not in ids   # only text registered


class TestText:
    def test_uppercase(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-uppercase", 1, bindings={"text": Static(value="hello")})],
        )
        assert r["n"]["result"] == "HELLO"

    def test_lowercase(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-lowercase", 1, bindings={"text": Static(value="HELLO")})],
        )
        assert r["n"]["result"] == "hello"

    def test_trim(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-trim", 1, bindings={"text": Static(value="  hi  ")})],
        )
        assert r["n"]["result"] == "hi"

    def test_length(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-length", 1, bindings={"text": Static(value="hello")})],
        )
        assert r["n"]["result"] == 5

    def test_concat_with_separator(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-concat", 1, bindings={"a": Static(value="foo"), "b": Static(value="bar"), "separator": Static(value="-")})],
        )
        assert r["n"]["result"] == "foo-bar"

    def test_replace(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-replace", 1,
                       bindings={"text": Static(value="hello world"), "needle": Static(value="world"), "replacement": Static(value="there")})],
        )
        assert r["n"]["result"] == "hello there"

    def test_contains_case_insensitive(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-contains", 1,
                       bindings={"text": Static(value="Hello"), "needle": Static(value="hello"), "case_sensitive": Static(value=False)})],
        )
        assert r["n"]["result"] == Flag(True)

    def test_contains_case_sensitive(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-contains", 1,
                       bindings={"text": Static(value="Hello"), "needle": Static(value="hello"), "case_sensitive": Static(value=True)})],
        )
        assert r["n"]["result"] == Flag(False)

    def test_split(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-split", 1, bindings={"text": Static(value="a,b,c"), "separator": Static(value=",")})],
        )
        assert r["n"]["result"] == ["a", "b", "c"]

    def test_join(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-join", 1, bindings={"parts": Static(value=["a", "b", "c"]), "separator": Static(value="-")})],
        )
        assert r["n"]["result"] == "a-b-c"

    def test_reverse(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-reverse", 1, bindings={"text": Static(value="hello")})],
        )
        assert r["n"]["result"] == "olleh"


class TestMath:
    def test_add(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-add", 1, bindings={"a": Static(value=2), "b": Static(value=3)})],
        )
        assert r["n"]["result"] == 5

    def test_subtract(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-subtract", 1, bindings={"a": Static(value=5), "b": Static(value=3)})],
        )
        assert r["n"]["result"] == 2

    def test_multiply(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-multiply", 1, bindings={"a": Static(value=4), "b": Static(value=3)})],
        )
        assert r["n"]["result"] == 12

    def test_divide(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-divide", 1, bindings={"a": Static(value=10), "b": Static(value=4)})],
        )
        assert r["n"]["result"] == 2.5

    def test_divide_by_zero_raises(self, full_registry):
        with pytest.raises(FlowExecutionError):
            _run(
                full_registry,
                [GraphNode("n", "math-divide", 1, bindings={"a": Static(value=1), "b": Static(value=0)})],
            )

    def test_modulo(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-modulo", 1, bindings={"a": Static(value=10), "b": Static(value=3)})],
        )
        assert r["n"]["result"] == 1

    def test_round_default(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-round", 1, bindings={"value": Static(value=3.7)})],
        )
        assert r["n"]["result"] == 4

    def test_round_to_decimals(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-round", 1, bindings={"value": Static(value=3.14159), "decimals": Static(value=2)})],
        )
        assert r["n"]["result"] == 3.14

    def test_min(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-min", 1, bindings={"values": Static(value=[3, 1, 2])})],
        )
        assert r["n"]["result"] == 1

    def test_max(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-max", 1, bindings={"values": Static(value=[3, 1, 2])})],
        )
        assert r["n"]["result"] == 3

    def test_abs(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-abs", 1, bindings={"value": Static(value=-7)})],
        )
        assert r["n"]["result"] == 7


class TestLogic:
    """The two ``if`` nodes route text to one named output and ``SKIPPED`` on the other."""

    def test_if_empty_routes_to_empty_branch(self, full_registry):
        r = _run(
            full_registry,
            [
                GraphNode("cond", "logic-if-empty", 1, bindings={"text": Static(value="   ")}),
                GraphNode("down", "text-uppercase", 1, bindings={"text": Sources(refs=(Ref('cond', 'empty'),))}),
            ],
        )
        # Empty branch delivered "   " to the downstream node
        assert r["down"]["result"] == "   "

    def test_if_empty_routes_to_non_empty_branch(self, full_registry):
        # The "empty" branch consumer should be skipped.
        r = _run(
            full_registry,
            [
                GraphNode("cond", "logic-if-empty", 1, bindings={"text": Static(value="hi")}),
                GraphNode("up", "text-uppercase", 1, bindings={"text": Sources(refs=(Ref('cond', 'not_empty'),))}),
                GraphNode("other", "text-uppercase", 1, bindings={"text": Sources(refs=(Ref('cond', 'empty'),))}),
            ],
        )
        assert r["up"]["result"] == "HI"
        assert "other" not in r   # skipped

    def test_if_equals_true(self, full_registry):
        r = _run(
            full_registry,
            [
                GraphNode("cond", "logic-if-equals", 1, bindings={"a": Static(value="foo"), "b": Static(value="foo")}),
                GraphNode("eq", "text-uppercase", 1, bindings={"text": Sources(refs=(Ref('cond', 'equal'),))}),
            ],
        )
        assert r["eq"]["result"] == "FOO"

    def test_if_equals_case_insensitive(self, full_registry):
        r = _run(
            full_registry,
            [
                GraphNode("cond", "logic-if-equals", 1,
                          bindings={"a": Static(value="Foo"), "b": Static(value="FOO"), "case_sensitive": Static(value=False)}),
                GraphNode("eq", "text-uppercase", 1, bindings={"text": Sources(refs=(Ref('cond', 'equal'),))}),
            ],
        )
        assert r["eq"]["result"] == "FOO"

    def test_not(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "logic-not", 1, bindings={"value": Static(value=True)})],
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
            [GraphNode("n", "json-parse", 1, bindings={"text": Static(value='{"a": 1, "b": [2, 3]}')})],
        )
        assert r["n"]["result"] == Json({"a": 1, "b": [2, 3]})

    def test_stringify(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "json-stringify", 1,
                       bindings={"value": Static(value={"b": 2, "a": 1}), "sort_keys": Static(value=True)})],
        )
        assert r["n"]["result"] == '{"a": 1, "b": 2}'

    def test_stringify_indented(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "json-stringify", 1, bindings={"value": Static(value={"x": 1}), "indent": Static(value=2)})],
        )
        assert "\n" in r["n"]["result"]

    def test_get_simple_key(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "json-get", 1,
                       bindings={"value": Static(value={"user": {"name": "Ada"}}), "path": Static(value="user.name")})],
        )
        assert r["n"]["result"] == Json("Ada")

    def test_get_list_index(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "json-get", 1,
                       bindings={"value": Static(value={"items": [{"id": "a"}, {"id": "b"}]}), "path": Static(value="items.1.id")})],
        )
        assert r["n"]["result"] == Json("b")

    def test_get_missing_path_returns_none(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "json-get", 1,
                       bindings={"value": Static(value={"a": 1}), "path": Static(value="does.not.exist")})],
        )
        assert r["n"]["result"] == Json(None)


class TestRegex:
    def test_match_true(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "regex-match", 1,
                       bindings={"text": Static(value="hello 123 world"), "pattern": Static(value=r"\d+")})],
        )
        assert r["n"]["result"] == Flag(True)

    def test_match_false(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "regex-match", 1,
                       bindings={"text": Static(value="nothing numeric"), "pattern": Static(value=r"\d+")})],
        )
        assert r["n"]["result"] == Flag(False)

    def test_match_ignore_case(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "regex-match", 1,
                       bindings={"text": Static(value="Hello"), "pattern": Static(value=r"hello"), "ignore_case": Static(value=True)})],
        )
        assert r["n"]["result"] == Flag(True)

    def test_replace(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "regex-replace", 1,
                       bindings={"text": Static(value="a1b2c3"), "pattern": Static(value=r"\d"), "replacement": Static(value="-")})],
        )
        assert r["n"]["result"] == "a-b-c-"

    def test_extract_findall_when_no_groups(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "regex-extract", 1,
                       bindings={"text": Static(value="a1 b22 c333"), "pattern": Static(value=r"\d+")})],
        )
        assert r["n"]["result"] == ["1", "22", "333"]

    def test_extract_uses_first_group(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "regex-extract", 1,
                       bindings={"text": Static(value="name=Ada, age=36"), "pattern": Static(value=r"name=(\w+)")})],
        )
        assert r["n"]["result"] == ["Ada"]


class TestIntegration:
    """Several categories chained through edges."""

    def test_pipeline_split_trim_join_concat(self, full_registry):
        r = _run(
            full_registry,
            [
                GraphNode("src", "text-split", 1,
                          bindings={"text": Static(value=" a ,  b , c "), "separator": Static(value=",")}),
                # Reuse the split result, upper-cased after a join
                GraphNode("joined", "text-join", 1, bindings={"separator": Static(value="|"), "parts": Sources(refs=(Ref('src', 'result'),))}),
                GraphNode("upper", "text-uppercase", 1, bindings={"text": Sources(refs=(Ref('joined', 'result'),))}),
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
                GraphNode("a", "math-add", 1, bindings={"a": Static(value=2), "b": Static(value=3)}),     # 5
                GraphNode("b", "math-multiply", 1, bindings={"b": Static(value=4), "a": Sources(refs=(Ref('a', 'result'),))}),          # 5 * 4 = 20
                GraphNode("c", "math-round", 1, bindings={"decimals": Static(value=0), "value": Sources(refs=(Ref('b', 'result'),))}),      # 20
            ],
        )
        assert r["c"]["result"] == 20
