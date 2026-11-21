"""Widgets: the control a graph editor shows a person for an input.

A widget tells an editor how to draw an input, such as "show this as a dropdown". A node author gives an input a
widget on its ``Param``, inside ``Annotated``::

    @registry.node("shout", title="Shout", description="Capitals, and a mark.")
    def shout(line: str, mark: Annotated[str, Param(widget=Dropdown(), choices=["!", "?"])] = "!") -> str: ...

An editor then shows ``mark`` as a dropdown with the choices ``!`` and ``?``. The input's description, which an
editor reads, has the widget as JSON: ``{"type": "dropdown"}``.

Conductor itself never reads a widget, not when it compiles a graph and not when it runs one. A widget only says how
the control looks. Everything else about the input is on its ``Param``: its title, its description, its choices and
its limits, such as ``ge`` and ``le``. A value outside the limits is refused, whatever the widget.

An input with no widget is shown with the control the host's editor picks for its type, such as a line of text for a
``str``. So give an input a widget only when the editor should show something its type doesn't say: a ``Textarea``
for a long text, a ``Slider`` for a number in a range, or a ``DatePicker`` that starts at today. Conductor gives no
type a widget of its own, so an input without one has ``"widget": null`` in its description.

A widget is a frozen pydantic model. Each class writes out its ``type`` by hand: its name in lower case, with a
hyphen between words, so ``TextInput`` is ``"text-input"``.

Conductor has the common controls. A host adds its own by subclassing ``Widget``, and draws it with a component of
its own::

    class EntityDropdown(Widget):
        type: Literal["entity-dropdown"] = "entity-dropdown"
        entity_kind: str = ""
"""

from __future__ import annotations

from typing import Literal

from conductor._internal.base import ConductorModel


class Widget(ConductorModel):
    """The base class of every widget. A subclass declares its ``type`` as a ``Literal``, and any settings it has.

    ``Widget`` itself declares no ``type``. That way a subclass's ``type`` overrides nothing, and a strict type
    checker accepts it.
    """


class TextInput(Widget):
    """A single line of text."""

    type: Literal["text-input"] = "text-input"


class Textarea(Widget):
    """A box for text over several lines. Its ``rows`` sets how tall it is.

    Attributes:
        rows: How many lines of text the box shows before it scrolls.
    """

    type: Literal["textarea"] = "textarea"
    rows: int = 4


class Dropdown(Widget):
    """A list to pick one value from. The values are the input's ``Param(choices=...)``, or those of its ``Literal``
    type."""

    type: Literal["dropdown"] = "dropdown"


class Slider(Widget):
    """A slider for a number. It runs from the ``ge`` to the ``le`` of the input's ``Param``.

    Attributes:
        step: How far one step of the slider moves the value, such as ``1`` for whole numbers. ``None`` lets the
            editor choose.
    """

    type: Literal["slider"] = "slider"
    step: float | None = None


class FileUpload(Widget):
    """A file a person uploads. An input whose type is a ``Series`` takes several files.

    Attributes:
        accept: The kinds of file the editor offers, written as for the ``accept`` of an HTML file input: file
            extensions (``".pdf"``) or media types (``"image/*"``). An empty tuple offers every kind.
        max_size_mb: The largest file the editor takes, in megabytes. ``None`` sets no limit in the editor.
    """

    type: Literal["file-upload"] = "file-upload"
    accept: tuple[str, ...] = ()
    max_size_mb: float | None = None


class NumberInput(Widget):
    """A box a person types a number in. It takes only whole numbers when the input is an ``int``.

    Attributes:
        step: How far the editor's up and down arrows move the value. ``None`` lets the editor choose.
    """

    type: Literal["number-input"] = "number-input"
    step: float | None = None


class Switch(Widget):
    """An on and off switch for a yes or no."""

    type: Literal["switch"] = "switch"


class DatePicker(Widget):
    """A calendar to pick a date from. With ``initial="today"``, it starts at the date the node is added.

    Attributes:
        initial: ``"today"`` asks the editor to fill in today's date when a person adds the node to a graph. A
            default can't do that, since it is fixed when the node is written. ``None`` leaves the input at its
            default.
    """

    type: Literal["date-picker"] = "date-picker"
    initial: Literal["today"] | None = None


class ListInput(Widget):
    """A list a person fills in one item at a time. The editor picks the control for each item from the items' type.

    That is why this widget has no settings.
    """

    type: Literal["list-input"] = "list-input"


# The reference (``python -m conductor.about``) lists the widgets in this order.
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
