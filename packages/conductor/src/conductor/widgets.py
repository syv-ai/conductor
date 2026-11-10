"""Widgets — how a person edits an input.

A widget is the control an input is edited with, plus what that control
needs: a slider's step, a text area's height. It is a frozen,
keyword-only record with a ``type``, and neither loading nor the engine
reads it: conductor carries it from the signature to whatever draws the
editor. An author writes one on the ``Param`` inside ``Annotated``::

    language: Annotated[Literal["en", "da"], Param(widget=Dropdown())]

The controls are named as an HTML form names them, and each one's ``type``
is its name in kebab case: ``TextInput`` is ``"text-input"``.

Conductor ships the common controls. A host adds its own by subclassing
``Widget`` beside the component that renders it::

    class EntityDropdown(Widget):
        type: Literal["entity-dropdown"] = "entity-dropdown"
        entity_kind: str = ""

A widget is the control and nothing more. What a person reads about the
input, whether an edge can reach it and the limits a value must keep to are
the ``Param``'s, where they are enforced. An input with no widget
gets no control in an editor: an edge fills it, or a value set in the graph. Nor does a widget change how the engine runs: a
pause is a node that raises ``Ask``, not a widget type.

Conductor picks no default widget for any type: a text may be a
textarea, a single line or a dropdown, so every input declares its own.
"""

from __future__ import annotations

from typing import Literal

from conductor._internal.conductor_model import ConductorModel


class Widget(ConductorModel):
    """The base of every control. Each control declares its own ``type: Literal[...]`` and what it needs; a host
    subclasses it for a control of its own the same way. The base declares no ``type``, so a subclass's
    ``Literal`` overrides nothing and passes a strict type checker."""


class TextInput(Widget):
    """Single-line text."""

    type: Literal["text-input"] = "text-input"


class Textarea(Widget):
    """Multi-line text."""

    type: Literal["textarea"] = "textarea"
    rows: int = 4


class Dropdown(Widget):
    """Pick one of the field's choices, which its record carries: ``Param(choices=)``, or a ``Literal``'s values."""

    type: Literal["dropdown"] = "dropdown"


class Slider(Widget):
    """A number picked on a slider, between the ``ge`` and ``le`` of its ``Param``."""

    type: Literal["slider"] = "slider"
    step: float | None = None


class FileUpload(Widget):
    """Files a person uploads: one, or a series of them when the field is a ``Series``."""

    type: Literal["file-upload"] = "file-upload"
    accept: tuple[str, ...] | None = None
    max_size_mb: float | None = None


class NumberInput(Widget):
    """A number typed in: a whole one when the field is an ``int``."""

    type: Literal["number-input"] = "number-input"
    step: float | None = None


class Switch(Widget):
    """A boolean, on or off."""

    type: Literal["switch"] = "switch"


class DatePicker(Widget):
    """A date picked from a calendar."""

    type: Literal["date-picker"] = "date-picker"
    #: ``"today"`` asks the editor to write today's date into the field
    #: when the node is added, a value no default could hold.
    initial: Literal["today"] | None = None


class ListInput(Widget):
    """A list of values typed by hand. The per-item control is derived by
    the host from the element type, so none is declared here."""

    type: Literal["list-input"] = "list-input"


__all__ = [
    "DatePicker",
    "Dropdown",
    "FileUpload",
    "ListInput",
    "NumberInput",
    "Slider",
    "Switch",
    "TextInput",
    "Textarea",
    "Widget",
]
