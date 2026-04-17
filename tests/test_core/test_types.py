"""Phase 1: Core types, enums, and sentinels."""

from conductor.types import NodeCategory, ResultFormat, WidgetType


class TestWidgetType:
    def test_standard_widgets_exist(self):
        assert WidgetType.TEXT == "text"
        assert WidgetType.TEXTAREA == "textarea"
        assert WidgetType.DROPDOWN == "dropdown"
        assert WidgetType.RANGE == "range"
        assert WidgetType.CHECKBOX == "checkbox"
        assert WidgetType.FILE == "file"
        assert WidgetType.OUTPUT == "output"

    def test_is_string_enum(self):
        assert isinstance(WidgetType.TEXT, str)
        assert WidgetType.TEXT == "text"


class TestResultFormat:
    def test_formats_exist(self):
        assert ResultFormat.SINGLE == "single"
        assert ResultFormat.MULTI == "multi"
        assert ResultFormat.DICT_SPREAD == "dict"


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
