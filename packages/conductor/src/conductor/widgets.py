"""Widgets: which control a graph editor shows a person for an input.

A node author gives an input a widget on its ``Param``, inside ``Annotated``::

    @registry.node("shout", title="Shout", description="Capitals, and a mark.")
    def shout(line: str, mark: Annotated[str, Param(widget=Dropdown(), choices=["!", "?"])] = "!") -> str: ...

An editor then shows ``mark`` as a dropdown with the choices ``!`` and ``?``. The input's description, which an
editor reads, has the widget as JSON: ``{"type": "dropdown"}``. A widget is a frozen pydantic model. Each class
writes its ``type`` by hand: its name in lower case, with a hyphen between words (``TextInput`` is
``"text-input"``).

Conductor itself never reads a widget, when it compiles a graph or runs it. A widget says only how an editor draws
the control: the input's title, description, choices and limits (such as ``ge`` and ``le``) are on its ``Param``,
and a value outside the limits is refused whatever the widget. An input with no widget has no control in an
editor; its value comes from an edge or from a constant in the graph record. No type has a default widget: a text
may be a text area, a single line or a dropdown, so an input that a person edits declares its own.

Conductor has the common controls. A host adds its own by subclassing ``Widget``, and draws it with its own
component::

    class EntityDropdown(Widget):
        type: Literal["entity-dropdown"] = "entity-dropdown"
        entity_kind: str = ""
"""

from __future__ import annotations

from typing import Literal

from conductor._internal.base import ConductorModel


class Widget(ConductorModel):
    """The base class of every widget. A subclass declares its ``type`` as a ``Literal`` and the settings it has.

    ``Widget`` itself declares no ``type``, so a subclass that declares one as a ``Literal`` overrides nothing,
    and a strict type checker accepts it."""


class TextInput(Widget):
    """A single line of text."""

    type: Literal["text-input"] = "text-input"


class Textarea(Widget):
    """Text over several lines, in a box ``rows`` lines high.

    Attributes:
        rows: How many lines of text the box shows before it scrolls.
    """

    type: Literal["textarea"] = "textarea"
    rows: int = 4


class Dropdown(Widget):
    """A choice of one value from a list: the input's ``Param(choices=...)``, or the values of its ``Literal`` type."""

    type: Literal["dropdown"] = "dropdown"


class Slider(Widget):
    """A number set with a slider, between the ``ge`` and the ``le`` of the input's ``Param``.

    Attributes:
        step: How far one step of the slider moves the value, such as ``1`` for whole numbers. ``None`` lets the
            editor choose.
    """

    type: Literal["slider"] = "slider"
    step: float | None = None


class FileUpload(Widget):
    """A file a person uploads, of a kind in ``accept`` and no larger than ``max_size_mb``. An input whose type is a
    ``Series`` takes several files.

    Attributes:
        accept: The kinds of file the editor offers, written as for the ``accept`` of an HTML file input: extensions
            (``".pdf"``) or media types (``"image/*"``). An empty tuple offers every kind.
        max_size_mb: The largest file, in megabytes, the editor takes. ``None`` sets no limit in the editor.
    """

    type: Literal["file-upload"] = "file-upload"
    accept: tuple[str, ...] = ()
    max_size_mb: float | None = None


class NumberInput(Widget):
    """A number a person types, a whole number when the input is an ``int``. Its arrows move it by ``step``.

    Attributes:
        step: How far the editor's up and down arrows move the value. ``None`` lets the editor choose.
    """

    type: Literal["number-input"] = "number-input"
    step: float | None = None


class Switch(Widget):
    """A switch for a boolean, on or off."""

    type: Literal["switch"] = "switch"


class DatePicker(Widget):
    """A date chosen from a calendar. ``initial="today"`` starts the input at the date the node is added.

    Attributes:
        initial: ``"today"`` asks the editor to set the input to today's date when a person adds the node to a
            graph. A default cannot say that, since it is fixed when the node is defined. ``None`` leaves the
            input at its default.
    """

    type: Literal["date-picker"] = "date-picker"
    initial: Literal["today"] | None = None


class ListInput(Widget):
    """A list a person enters one item at a time. The host's editor decides the control for an item from the type
    of the list's items, so this widget has no settings."""

    type: Literal["list-input"] = "list-input"


# The order in which the reference (``python -m conductor.about``) lists them.
__all__ = [
    "Widget",
    "TextInput",
    "Textarea",
    "Dropdown",
    "NumberInput",
    "Slider",
    "Switch",
    "DatePicker",
    "FileUpload",
    "ListInput",
]
