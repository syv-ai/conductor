"""Tests for ``NodeRegistry.merge`` and the surrounding composition story."""

from __future__ import annotations

from typing import Annotated

import pytest
from conductor import NodeRegistry
from conductor.widgets import Output, Text


def _node(reg: NodeRegistry, base_id: str, version: int, label: str) -> None:
    """Register a throwaway node with a unique body so identity is testable."""

    @reg.node(base_id, version=version, name=label, description=f"{label} desc")
    def _fn(x: Annotated[str, Text(label="x")]) -> Annotated[str, Output(label="y")]:
        return f"{label}:{x}"


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestMergeHappyPath:
    def test_clean_merge_pulls_every_node_across(self):
        a, b = NodeRegistry(), NodeRegistry()
        _node(a, "alpha", 1, "Alpha")
        _node(a, "beta", 1, "Beta")
        _node(b, "gamma", 1, "Gamma")

        a.merge(b)

        ids = {nd.id for nd in a.all()}
        assert ids == {"alpha@1", "beta@1", "gamma@1"}

    def test_returns_self_for_chaining(self):
        a, b, c = NodeRegistry(), NodeRegistry(), NodeRegistry()
        _node(b, "x", 1, "X")
        _node(c, "y", 1, "Y")

        # Chain two merges on one line.
        result = a.merge(b).merge(c)

        assert result is a
        assert {nd.id for nd in a.all()} == {"x@1", "y@1"}

    def test_source_registry_is_not_mutated(self):
        a, b = NodeRegistry(), NodeRegistry()
        _node(b, "x", 1, "X")
        _node(b, "y", 1, "Y")
        before = {nd.id for nd in b.all()}

        a.merge(b)

        after = {nd.id for nd in b.all()}
        assert before == after
        # And the nodes in a are the same objects (NodeDefinition is frozen,
        # so sharing is safe).
        assert a.get("x@1") is b.get("x@1")

    def test_node_metadata_preserved(self):
        a, b = NodeRegistry(), NodeRegistry()
        _node(b, "x", 1, "Example")

        a.merge(b)

        merged = a.get("x@1")
        original = b.get("x@1")
        assert merged.name == original.name == "Example"
        assert merged.description == original.description
        assert merged.inputs == original.inputs
        assert merged.outputs == original.outputs

    def test_versions_coexist_after_merge(self):
        """v1 from left + v2 from right both survive; v1 becomes deprecated."""
        a, b = NodeRegistry(), NodeRegistry()
        _node(a, "foo", 1, "Foo v1")
        _node(b, "foo", 2, "Foo v2")

        a.merge(b)

        assert a.get("foo@1") is not None
        assert a.get("foo@2") is not None
        assert a.get_latest("foo").version == 2
        assert a.is_deprecated("foo@1") is True
        assert a.is_deprecated("foo@2") is False

    def test_merge_preserves_latest_tracking(self):
        """When both sides hold non-conflicting versions, `get_latest` sees
        the highest after merge."""
        a, b = NodeRegistry(), NodeRegistry()
        _node(a, "foo", 2, "v2")
        _node(b, "foo", 3, "v3")

        a.merge(b)

        assert a.get_latest("foo").version == 3

    def test_empty_source_leaves_target_unchanged(self):
        a = NodeRegistry()
        _node(a, "x", 1, "X")
        ids_before = {nd.id for nd in a.all()}

        a.merge(NodeRegistry())

        assert {nd.id for nd in a.all()} == ids_before

    def test_merge_into_empty_target(self):
        a, b = NodeRegistry(), NodeRegistry()
        _node(b, "x", 1, "X")
        _node(b, "y", 1, "Y")

        a.merge(b)

        assert {nd.id for nd in a.all()} == {"x@1", "y@1"}


# ---------------------------------------------------------------------------
# Conflict modes
# ---------------------------------------------------------------------------


class TestMergeConflictModes:
    def test_conflict_raises_by_default(self):
        a, b = NodeRegistry(), NodeRegistry()
        _node(a, "dup", 1, "A-version")
        _node(b, "dup", 1, "B-version")

        with pytest.raises(ValueError) as excinfo:
            a.merge(b)

        msg = str(excinfo.value)
        assert "dup@1" in msg
        assert "on_conflict='skip'" in msg       # actionable hint
        assert "bump `version`" in msg           # actionable hint

    def test_conflict_skip_keeps_existing_and_ignores_incoming(self):
        a, b = NodeRegistry(), NodeRegistry()
        _node(a, "dup", 1, "A-version")
        _node(b, "dup", 1, "B-version")

        result = a.merge(b, on_conflict="skip")

        # self returned
        assert result is a
        # And the *existing* node stayed
        assert a.get("dup@1").name == "A-version"

    def test_conflict_skip_still_copies_non_conflicting_nodes(self):
        a, b = NodeRegistry(), NodeRegistry()
        _node(a, "dup", 1, "A")
        _node(b, "dup", 1, "B")
        _node(b, "fresh", 1, "Fresh")

        a.merge(b, on_conflict="skip")

        assert a.get("dup@1").name == "A"
        assert a.get("fresh@1") is not None      # non-conflicting merged normally

    def test_error_summary_collects_all_conflicts(self):
        a, b = NodeRegistry(), NodeRegistry()
        _node(a, "x", 1, "A-x")
        _node(a, "y", 1, "A-y")
        _node(b, "x", 1, "B-x")
        _node(b, "y", 1, "B-y")
        _node(b, "z", 1, "B-z")

        with pytest.raises(ValueError) as excinfo:
            a.merge(b, on_conflict="error-summary")

        msg = str(excinfo.value)
        assert "2 conflict(s)" in msg
        assert "x@1" in msg
        assert "y@1" in msg
        # The summary raises *after* trying to copy, so the non-conflicting
        # z@1 will have already been added. This is intentional: the summary
        # mode is diagnostic; users who want atomicity should pre-check or
        # fall back to the default raise mode.
        assert a.get("z@1") is not None

    def test_unknown_on_conflict_mode_raises(self):
        a, b = NodeRegistry(), NodeRegistry()
        _node(b, "x", 1, "X")
        with pytest.raises(ValueError, match="Unknown on_conflict mode"):
            a.merge(b, on_conflict="clobber")

    def test_conflict_scope_is_full_id_not_base_id(self):
        """foo@1 in left and foo@2 in right is NOT a conflict."""
        a, b = NodeRegistry(), NodeRegistry()
        _node(a, "foo", 1, "A-foo-v1")
        _node(b, "foo", 2, "B-foo-v2")

        # Should succeed with the strict default
        a.merge(b)

        assert {nd.version for nd in a._by_base_id["foo"]} == {1, 2}


# ---------------------------------------------------------------------------
# get_default_registry (from conductor_nodes)
# ---------------------------------------------------------------------------


class TestDefaultRegistryFactory:
    def test_returns_populated_registry(self):
        import conductor_nodes

        reg = conductor_nodes.get_default_registry()
        ids = {nd.id for nd in reg.all()}
        # One spot-check per category
        assert "text-uppercase@1" in ids
        assert "math-add@1" in ids
        assert "logic-if-empty@1" in ids
        assert "for-each-start@1" in ids
        assert "json-parse@1" in ids
        assert "regex-match@1" in ids

    def test_returns_fresh_registry_each_call(self):
        """Mutating one shouldn't affect the next call — no shared state."""
        import conductor_nodes

        a = conductor_nodes.get_default_registry()
        # Trash `a` by adding a custom node — next call should not see it.
        @a.node("user-only", version=1, name="U", description="x")
        def _user_node(x: Annotated[str, Text(label="x")]) -> Annotated[str, Output(label="y")]:
            return x

        b = conductor_nodes.get_default_registry()
        assert "user-only@1" in {nd.id for nd in a.all()}
        assert "user-only@1" not in {nd.id for nd in b.all()}

    def test_categories_filter_respected(self):
        import conductor_nodes

        reg = conductor_nodes.get_default_registry(categories=["text"])
        ids = {nd.id for nd in reg.all()}
        assert "text-uppercase@1" in ids
        assert "math-add@1" not in ids
        assert "for-each-start@1" not in ids

    def test_merges_cleanly_into_a_user_registry(self):
        """The intended use case — compose default nodes with your own."""
        import conductor_nodes

        mine = NodeRegistry()
        _node(mine, "my-node", 1, "Mine")

        mine.merge(conductor_nodes.get_default_registry())

        ids = {nd.id for nd in mine.all()}
        assert "my-node@1" in ids
        assert "text-uppercase@1" in ids
        assert "math-add@1" in ids
