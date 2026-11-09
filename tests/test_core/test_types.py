"""Phase 1: Core types, enums, and sentinels."""

class TestSentinel:
    def test_skipped_is_singleton(self):
        from conductor.skip_and_ask import SKIPPED

        assert SKIPPED is SKIPPED
        assert repr(SKIPPED) == "SKIPPED"

    def test_skipped_is_falsy(self):
        from conductor.skip_and_ask import SKIPPED

        assert not SKIPPED
