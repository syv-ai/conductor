"""Pins for the public typed registry serialization and the widget-key vocabulary."""

import dataclasses

from conductor import widgets as widgets_module
from conductor.registry.schema import serialize_input, serialize_node, serialize_output
from conductor.widgets import WIDGET_SCHEMA_KEYS, FileUpload, Widget

# Keys every widget's to_schema() emits that are input-level, not
# widget-config-level (the registry strips these before storing
# widget_config — see registry/__init__.py).
_BASE_KEYS = {"widget", "label", "description", "disable_handle"}


def _dummy_value(field_name: str):
    """A plausible, type-correct non-None value for an optional widget config field.

    The factory stuffs every optional field so ``to_schema()`` emits every key
    it can. The values must be type-correct (not just present): the emitted
    schema is validated against ``SerializedInput`` — the typed wire model — so
    a str where an int belongs would be a false failure, not a real one.
    """
    if field_name == "item_widget":
        return None  # nested Widget slot — leave unset; key emits as null anyway
    if field_name in ("choices", "variables"):
        return ["x"]
    if field_name == "accept":
        return [".pdf"]
    if field_name == "choices_map":
        return {"a": ["x"]}
    if field_name == "hidden_when":
        return {"field": ["value"]}
    if field_name == "schema":
        return {"k": "v"}
    if field_name in (
        "min_length",
        "max_length",
        "min_items",
        "max_items",
        "min_selected",
        "max_selected",
        "min_rows",
        "min_columns",
        "min_val",
        "max_val",
        "step",
        "max_size_mb",
        "rows",
    ):
        return 1
    if field_name in ("multiple", "integer_only", "human_review"):
        return True
    return "x"


def _all_widget_instances() -> list[Widget]:
    """One maximally-populated instance per concrete Widget subclass.

    Optional (None-defaulted) config fields are set to dummy values so
    ``to_schema()`` emits every key the widget can possibly emit.
    """
    instances: list[Widget] = []
    for obj in vars(widgets_module).values():
        if (
            isinstance(obj, type)
            and issubclass(obj, Widget)
            and obj is not Widget
            and not getattr(obj, "__abstractmethods__", None)
        ):
            widget = obj(label="x")
            for f in dataclasses.fields(obj):
                if f.name in ("label", "description", "disable_handle"):
                    continue
                current = getattr(widget, f.name, None)
                if current is None:
                    dummy = _dummy_value(f.name)
                    if dummy is not None:
                        setattr(widget, f.name, dummy)
                elif current is False:
                    # Bool-gated keys (``advanced`` &c.) only emit when True.
                    setattr(widget, f.name, True)
            instances.append(widget)
    return instances


def test_widget_schema_keys_covers_every_emitted_key() -> None:
    """WIDGET_SCHEMA_KEYS is the complete vocabulary of widget_config keys.

    A widget emitting a key missing from the constant means a host that
    types its port model against WIDGET_SCHEMA_KEYS silently loses that
    key — exactly the drop this constant exists to prevent.
    """
    emitted: set[str] = set()
    for widget in _all_widget_instances():
        emitted |= set(widget.to_schema()) - _BASE_KEYS
    missing = emitted - WIDGET_SCHEMA_KEYS
    assert not missing, f"widget keys missing from WIDGET_SCHEMA_KEYS: {sorted(missing)}"


def test_widget_schema_keys_has_no_unemittable_entries() -> None:
    """The constant carries no speculative keys no stdlib widget can emit.

    Hosts may still receive extra keys from custom widgets; this pin only
    keeps the stdlib vocabulary honest in both directions.
    """
    emitted: set[str] = set()
    for widget in _all_widget_instances():
        emitted |= set(widget.to_schema()) - _BASE_KEYS
    stale = WIDGET_SCHEMA_KEYS - emitted
    assert not stale, f"WIDGET_SCHEMA_KEYS entries no widget emits: {sorted(stale)}"


def test_serialize_functions_are_public() -> None:
    assert callable(serialize_input)
    assert callable(serialize_output)
    assert callable(serialize_node)


def test_fileupload_accept_serializes_as_list() -> None:
    """Canonical wire shape: ``accept`` is always a list of extensions."""
    assert FileUpload(label="Fil", accept=".pdf").to_schema()["accept"] == [".pdf"]
    assert FileUpload(label="Fil", accept=[".pdf", ".csv"]).to_schema()["accept"] == [
        ".pdf",
        ".csv",
    ]
    assert "accept" not in FileUpload(label="Fil").to_schema()
