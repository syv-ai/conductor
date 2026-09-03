"""Phase 1: Core types, enums, and sentinels."""

from conductor.types import NodeCategory


class TestNodeCategory:
    def test_categories_exist(self):
        assert NodeCategory.IO == "io"
        assert NodeCategory.CONTROL == "control"


class TestSentinel:
    def test_skipped_is_singleton(self):
        from conductor._sentinel import SKIPPED

        assert SKIPPED is SKIPPED
        assert repr(SKIPPED) == "SKIPPED"

    def test_skipped_is_falsy(self):
        from conductor._sentinel import SKIPPED

        assert not SKIPPED
