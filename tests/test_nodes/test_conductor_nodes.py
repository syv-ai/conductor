"""End-to-end tests for the conductor-nodes library.

Each node is exercised through ``compile`` + ``execute_sync`` so we verify
registration, widget annotations, and runtime behavior together.
"""

from __future__ import annotations

import conductor_nodes
import pytest
from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor.compound.for_each import FOR_EACH
from conductor.execution.engine import execute_sync

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def full_registry() -> NodeRegistry:
    reg = NodeRegistry()
    conductor_nodes.register_all(reg)
    return reg


def _run(reg: NodeRegistry, nodes, edges, **compile_kwargs):
    compiled = compile(nodes=nodes, edges=edges, registry=reg, **compile_kwargs)
    return execute_sync(compiled)


# ---------------------------------------------------------------------------
# Package-level surface
# ---------------------------------------------------------------------------


class TestPackageSurface:
    def test_register_all_registers_every_category(self, full_registry):
        ids = {nd.id for nd in full_registry.all()}
        # Spot-check one node per category
        assert "text-uppercase@1" in ids
        assert "math-add@1" in ids
        assert "logic-if-empty@1" in ids
        assert "for-each-start@1" in ids
        assert "json-parse@1" in ids
        assert "regex-match@1" in ids

    def test_register_all_respects_categories_filter(self):
        reg = NodeRegistry()
        conductor_nodes.register_all(reg, categories=["text", "math"])
        ids = {nd.id for nd in reg.all()}
        assert "text-uppercase@1" in ids
        assert "math-add@1" in ids
        assert "json-parse@1" not in ids
        assert "logic-if-empty@1" not in ids

    def test_register_all_rejects_unknown_category(self):
        reg = NodeRegistry()
        with pytest.raises(KeyError, match="Unknown category"):
            conductor_nodes.register_all(reg, categories=["doesnt-exist"])

    def test_individual_modules_expose_register(self):
        reg = NodeRegistry()
        conductor_nodes.text.register(reg)
        ids = {nd.id for nd in reg.all()}
        assert "text-uppercase@1" in ids
        assert "math-add@1" not in ids   # only text registered


# ---------------------------------------------------------------------------
# Text
# ---------------------------------------------------------------------------


class TestText:
    def test_uppercase(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-uppercase@1", {"text": "hello"})], [],
        )
        assert r["n"]["result"] == "HELLO"

    def test_lowercase(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-lowercase@1", {"text": "HELLO"})], [],
        )
        assert r["n"]["result"] == "hello"

    def test_trim(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-trim@1", {"text": "  hi  "})], [],
        )
        assert r["n"]["result"] == "hi"

    def test_length(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-length@1", {"text": "hello"})], [],
        )
        assert r["n"]["result"] == 5

    def test_concat_with_separator(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-concat@1", {"a": "foo", "b": "bar", "separator": "-"})],
            [],
        )
        assert r["n"]["result"] == "foo-bar"

    def test_replace(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-replace@1",
                       {"text": "hello world", "needle": "world", "replacement": "there"})],
            [],
        )
        assert r["n"]["result"] == "hello there"

    def test_contains_case_insensitive(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-contains@1",
                       {"text": "Hello", "needle": "hello", "case_sensitive": False})],
            [],
        )
        assert r["n"]["result"] is True

    def test_contains_case_sensitive(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-contains@1",
                       {"text": "Hello", "needle": "hello", "case_sensitive": True})],
            [],
        )
        assert r["n"]["result"] is False

    def test_split(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-split@1", {"text": "a,b,c", "separator": ","})],
            [],
        )
        assert r["n"]["result"] == ["a", "b", "c"]

    def test_join(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-join@1", {"parts": ["a", "b", "c"], "separator": "-"})],
            [],
        )
        assert r["n"]["result"] == "a-b-c"

    def test_reverse(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "text-reverse@1", {"text": "hello"})], [],
        )
        assert r["n"]["result"] == "olleh"


# ---------------------------------------------------------------------------
# Math
# ---------------------------------------------------------------------------


class TestMath:
    def test_add(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-add@1", {"a": 2, "b": 3})], [],
        )
        assert r["n"]["result"] == 5

    def test_subtract(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-subtract@1", {"a": 5, "b": 3})], [],
        )
        assert r["n"]["result"] == 2

    def test_multiply(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-multiply@1", {"a": 4, "b": 3})], [],
        )
        assert r["n"]["result"] == 12

    def test_divide(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-divide@1", {"a": 10, "b": 4})], [],
        )
        assert r["n"]["result"] == 2.5

    def test_divide_by_zero_raises(self, full_registry):
        from conductor.errors import FlowExecutionError
        with pytest.raises(FlowExecutionError):
            _run(
                full_registry,
                [GraphNode("n", "math-divide@1", {"a": 1, "b": 0})], [],
            )

    def test_modulo(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-modulo@1", {"a": 10, "b": 3})], [],
        )
        assert r["n"]["result"] == 1

    def test_round_default(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-round@1", {"value": 3.7})], [],
        )
        assert r["n"]["result"] == 4

    def test_round_to_decimals(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-round@1", {"value": 3.14159, "decimals": 2})], [],
        )
        assert r["n"]["result"] == 3.14

    def test_min(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-min@1", {"values": [3, 1, 2]})], [],
        )
        assert r["n"]["result"] == 1

    def test_max(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-max@1", {"values": [3, 1, 2]})], [],
        )
        assert r["n"]["result"] == 3

    def test_abs(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "math-abs@1", {"value": -7})], [],
        )
        assert r["n"]["result"] == 7


# ---------------------------------------------------------------------------
# Logic (branching via SKIPPED)
# ---------------------------------------------------------------------------


class TestLogic:
    def test_if_empty_routes_to_empty_branch(self, full_registry):
        r = _run(
            full_registry,
            [
                GraphNode("cond", "logic-if-empty@1", {"text": "   "}),
                GraphNode("down", "text-uppercase@1", None),
            ],
            [GraphEdge("e1", "cond", "down", "output_2", "text")],
        )
        # Empty branch delivered "   " to the downstream node
        assert r["down"]["result"] == "   "

    def test_if_empty_routes_to_non_empty_branch(self, full_registry):
        # The "empty" branch consumer should be skipped.
        r = _run(
            full_registry,
            [
                GraphNode("cond", "logic-if-empty@1", {"text": "hi"}),
                GraphNode("up", "text-uppercase@1", None),
                GraphNode("other", "text-uppercase@1", None),
            ],
            [
                GraphEdge("e1", "cond", "up", "output_1", "text"),
                GraphEdge("e2", "cond", "other", "output_2", "text"),
            ],
        )
        assert r["up"]["result"] == "HI"
        assert "other" not in r   # skipped

    def test_if_equals_true(self, full_registry):
        r = _run(
            full_registry,
            [
                GraphNode("cond", "logic-if-equals@1", {"a": "foo", "b": "foo"}),
                GraphNode("eq", "text-uppercase@1", None),
            ],
            [GraphEdge("e1", "cond", "eq", "output_1", "text")],
        )
        assert r["eq"]["result"] == "FOO"

    def test_if_equals_case_insensitive(self, full_registry):
        r = _run(
            full_registry,
            [
                GraphNode("cond", "logic-if-equals@1",
                          {"a": "Foo", "b": "FOO", "case_sensitive": False}),
                GraphNode("eq", "text-uppercase@1", None),
            ],
            [GraphEdge("e1", "cond", "eq", "output_1", "text")],
        )
        assert r["eq"]["result"] == "FOO"

    def test_not(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "logic-not@1", {"value": True})], [],
        )
        assert r["n"]["result"] is False


# ---------------------------------------------------------------------------
# Loop markers (through the FOR_EACH compound)
# ---------------------------------------------------------------------------


class TestLoop:
    def test_for_each_collects_upper_of_each_item(self, full_registry):
        r = _run(
            full_registry,
            [
                GraphNode("start", "for-each-start@1", {"items": ["a", "b", "c"]}),
                GraphNode("body", "text-uppercase@1", None),
                GraphNode("end", "for-each-end@1", None),
            ],
            [
                GraphEdge("e1", "start", "body", "output_1", "text"),
                GraphEdge("e2", "body", "end", "result", "item"),
            ],
            compound_types=[FOR_EACH],
        )
        assert r["end"]["result"] == ["A", "B", "C"]


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------


class TestJSON:
    def test_parse(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "json-parse@1", {"text": '{"a": 1, "b": [2, 3]}'})], [],
        )
        assert r["n"]["result"] == {"a": 1, "b": [2, 3]}

    def test_stringify(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "json-stringify@1",
                       {"value": {"b": 2, "a": 1}, "sort_keys": True})],
            [],
        )
        assert r["n"]["result"] == '{"a": 1, "b": 2}'

    def test_stringify_indented(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "json-stringify@1", {"value": {"x": 1}, "indent": 2})],
            [],
        )
        assert "\n" in r["n"]["result"]

    def test_get_simple_key(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "json-get@1",
                       {"value": {"user": {"name": "Ada"}}, "path": "user.name"})],
            [],
        )
        assert r["n"]["result"] == "Ada"

    def test_get_list_index(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "json-get@1",
                       {"value": {"items": [{"id": "a"}, {"id": "b"}]},
                        "path": "items.1.id"})],
            [],
        )
        assert r["n"]["result"] == "b"

    def test_get_missing_path_returns_none(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "json-get@1",
                       {"value": {"a": 1}, "path": "does.not.exist"})],
            [],
        )
        assert r["n"]["result"] is None


# ---------------------------------------------------------------------------
# Regex
# ---------------------------------------------------------------------------


class TestRegex:
    def test_match_true(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "regex-match@1",
                       {"text": "hello 123 world", "pattern": r"\d+"})],
            [],
        )
        assert r["n"]["result"] is True

    def test_match_false(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "regex-match@1",
                       {"text": "nothing numeric", "pattern": r"\d+"})],
            [],
        )
        assert r["n"]["result"] is False

    def test_match_ignore_case(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "regex-match@1",
                       {"text": "Hello", "pattern": r"hello", "ignore_case": True})],
            [],
        )
        assert r["n"]["result"] is True

    def test_replace(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "regex-replace@1",
                       {"text": "a1b2c3", "pattern": r"\d", "replacement": "-"})],
            [],
        )
        assert r["n"]["result"] == "a-b-c-"

    def test_extract_findall_when_no_groups(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "regex-extract@1",
                       {"text": "a1 b22 c333", "pattern": r"\d+"})],
            [],
        )
        assert r["n"]["result"] == ["1", "22", "333"]

    def test_extract_uses_first_group(self, full_registry):
        r = _run(
            full_registry,
            [GraphNode("n", "regex-extract@1",
                       {"text": "name=Ada, age=36", "pattern": r"name=(\w+)"})],
            [],
        )
        assert r["n"]["result"] == ["Ada"]


# ---------------------------------------------------------------------------
# Integration — chain multiple categories together
# ---------------------------------------------------------------------------


class TestIntegration:
    def test_pipeline_split_trim_join_concat(self, full_registry):
        r = _run(
            full_registry,
            [
                GraphNode("src", "text-split@1",
                          {"text": " a ,  b , c ", "separator": ","}),
                # Reuse the split result, upper-cased after a join
                GraphNode("joined", "text-join@1", {"separator": "|"}),
                GraphNode("upper", "text-uppercase@1", None),
            ],
            [
                GraphEdge("e1", "src", "joined", "result", "parts"),
                GraphEdge("e2", "joined", "upper", "result", "text"),
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
                GraphNode("a", "math-add@1", {"a": 2, "b": 3}),     # 5
                GraphNode("b", "math-multiply@1", {"b": 4}),          # 5 * 4 = 20
                GraphNode("c", "math-round@1", {"decimals": 0}),      # 20
            ],
            [
                GraphEdge("e1", "a", "b", "result", "a"),
                GraphEdge("e2", "b", "c", "result", "value"),
            ],
        )
        assert r["c"]["result"] == 20
