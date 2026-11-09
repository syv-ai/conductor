"""Widgets — how a person edits an input.

A widget is the control an input is edited with, plus what that control
needs: a dropdown's choices, a text area's height. It is a frozen,
keyword-only record with a ``type``, and neither compile nor the engine
reads it: conductor carries it from the signature to whatever draws the
editor. An author writes one on the ``Param`` inside ``Annotated``::

    language: Annotated[Text, Param(widget=Dropdown(choices=(Choice(id="en", title="English"),)))]

Conductor ships the common controls. A host adds its own by subclassing
``Widget`` beside the component that renders it::

    class EntityDropdown(Widget):
        type: Literal["entity-dropdown"] = "entity-dropdown"
        entity_kind: str = ""

A widget is the control and nothing more. What a person reads about the
input, whether an edge can reach it and the limits a value must keep to are
the ``Param``'s, where they are enforced. An input with no widget is
filled by an edge only. Nor does a widget change how the engine runs: a
pause is a node that returns ``AskPerson``, not a widget type.

Conductor picks no default widget for any type: a text may be a
textarea, a single line or a dropdown, so every input declares its own.
The three controls whose plain names collide with a host's types are
named for what they are — ``TextWidget``, ``NumberWidget``,
``ListWidget`` — so ``from conductor.widgets import TextWidget`` sits
beside a host's ``Text`` without an alias.
"""

from __future__ import annotations

from typing import Literal

from conductor._internal.conductor_model import ConductorModel


class Widget(ConductorModel):
    """The base of every control: a ``type`` and what that control needs. A host subclasses it for a control of its own."""

    type: str


class Choice(ConductorModel):
    """One option a person may pick: the ``id`` the value stores and the ``title`` shown."""

    id: str
    title: str


class TextWidget(Widget):
    """Single-line text."""

    type: Literal["text"] = "text"


class Textarea(Widget):
    """Multi-line text."""

    type: Literal["textarea"] = "textarea"
    rows: int = 4


class Dropdown(Widget):
    """Pick one of a declared vocabulary of ``Choice``s."""

    type: Literal["dropdown"] = "dropdown"
    choices: tuple[Choice, ...] = ()


class Range(Widget):
    """A number picked on a slider, between the ``ge`` and ``le`` of its ``Param``."""

    type: Literal["range"] = "range"
    step: float | None = None


class FileUpload(Widget):
    """Files a person uploads. ``multiple`` makes a series of them."""

    type: Literal["file"] = "file"
    accept: tuple[str, ...] | None = None
    max_size_mb: float | None = None
    multiple: bool = False


class NumberWidget(Widget):
    """A number typed in, optionally whole."""

    type: Literal["number"] = "number"
    step: float | None = None
    integer_only: bool = False


class Switch(Widget):
    """A boolean, on or off."""

    type: Literal["switch"] = "switch"


class DatePicker(Widget):
    """A date picked from a calendar."""

    type: Literal["datepicker"] = "datepicker"
    #: ``"today"`` asks the editor to write today's date into the field
    #: when the node is added. Declared here because "today" is not a
    #: constant a default could hold.
    seed: Literal["today"] | None = None


class ListWidget(Widget):
    """A list of values typed by hand. The per-item control is derived by
    the host from the element type, so none is declared here."""

    type: Literal["list"] = "list"


__all__ = [
    "Choice",
    "DatePicker",
    "Dropdown",
    "FileUpload",
    "ListWidget",
    "NumberWidget",
    "Range",
    "Switch",
    "TextWidget",
    "Textarea",
    "Widget",
]
