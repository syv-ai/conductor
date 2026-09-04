"""Widgets — how a person edits an input.

A widget is the control an input is edited with, plus what that control
needs: a dropdown's choices, a number's range. Every widget is a frozen,
keyword-only record with a ``kind`` discriminator, and ``AnyWidget`` is the
union of all of them, so pydantic dumps a widget and publishes a JSON
schema per kind. An author writes one inside ``Annotated`` on a ``run``
parameter::

    language: Annotated[Text, Dropdown(title="Language", choices=(Choice(id="en", title="English"),))]

Three things written on the widget belong to the field, not the control:
``title``, ``description`` and ``show_handle``. They sit on the widget
because a parameter has one annotation object; ``Interface.of`` copies
them onto the ``Input`` and they are left out of the widget's own dump, so
each travels once. Nothing downstream reads ``widget.title``.

A widget does not decide whether a cable can reach the input:
``show_handle`` defaults to ``True`` on the base and no control overrides
it. A node closes one input by writing ``show_handle=False`` on that
input's annotation. Nor does a widget change how the engine runs: a pause
is a node that returns ``Asks``, not a widget kind.

Conductor ships no default widget for any type: ``Text`` may be a
textarea, a single line or a dropdown, so every input declares its own.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

from pydantic import Discriminator, Field

#: Marks the three fields that belong to the ``Input``, not the control.
#: ``Interface.of`` copies them onto the ``Input``; they are left out of
#: the widget's own dump.
_Lifted = Field(exclude=True)


@dataclass(frozen=True, kw_only=True)
class Widget(ABC):
    """What every control has in common: a title, a description, and whether a cable can reach the field.

    Never subclassed outside this module: ``AnyWidget`` is built from the
    subclasses declared here, and a control the host's frontend cannot
    render is not a control.
    """

    title: Annotated[str, _Lifted]
    description: Annotated[str | None, _Lifted] = None
    show_handle: Annotated[bool, _Lifted] = True


@dataclass(frozen=True, kw_only=True)
class Choice:
    """One option a person may pick: the ``id`` the value stores and the ``title`` shown.

    A host declares them on the widget where it declares the input, so the
    vocabulary travels as data and no frontend list has to agree with a
    host table by hand. ``element`` is a dtype description when the option
    fits only one element type (a reduction that works on numbers), and
    ``None`` when it fits anything; an editor filters by it. Its sibling
    ``OperatorChoice`` does the same for a condition builder's operators.
    """

    id: str
    title: str
    element: dict | None = None


@dataclass(frozen=True, kw_only=True)
class OperatorChoice:
    """One operator a condition builder offers.

    The host's operator table, serialised onto ``IfElseBuilder`` so a
    frontend need not keep a copy. ``category`` files it under the host's
    value kinds; ``arity`` is 1 for an operator with no argument beside its
    operand ("is empty") and 2 for one with ("contains ...").
    """

    id: str
    title: str
    category: str
    arity: int


@dataclass(frozen=True, kw_only=True)
class Text(Widget):
    """Single-line text."""

    kind: Literal["text"] = "text"
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None


@dataclass(frozen=True, kw_only=True)
class Textarea(Widget):
    """Multi-line text."""

    kind: Literal["textarea"] = "textarea"
    min_length: int | None = None
    max_length: int | None = None
    rows: int = 4


@dataclass(frozen=True, kw_only=True)
class Dropdown(Widget):
    """Pick one of a declared vocabulary of ``Choice``s."""

    kind: Literal["dropdown"] = "dropdown"
    choices: tuple[Choice, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Range(Widget):
    """A number picked on a slider, between declared bounds."""

    kind: Literal["range"] = "range"
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None


@dataclass(frozen=True, kw_only=True)
class FileUpload(Widget):
    """Files a person uploads. ``multiple`` births a series of them."""

    kind: Literal["file"] = "file"
    accept: tuple[str, ...] | None = None
    max_size_mb: float | None = None
    multiple: bool = False


@dataclass(frozen=True, kw_only=True)
class ConnectionList(Widget):
    """Edited by wiring only: the value comes down a cable, so there is nothing to type.

    The control for a ``Series[X]`` input, an ``Any`` input and an open
    roster's rows.
    """

    kind: Literal["connection-list"] = "connection-list"


@dataclass(frozen=True, kw_only=True)
class Number(Widget):
    """A number typed in, optionally bounded and optionally whole."""

    kind: Literal["number"] = "number"
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    integer_only: bool = False


@dataclass(frozen=True, kw_only=True)
class Switch(Widget):
    """A boolean, on or off."""

    kind: Literal["switch"] = "switch"


@dataclass(frozen=True, kw_only=True)
class DatePicker(Widget):
    """A date picked from a calendar."""

    kind: Literal["datepicker"] = "datepicker"
    min_date: str | None = None
    max_date: str | None = None
    #: ``"today"`` asks the editor to write today's date into the field
    #: when the node is placed. Declared here because "today" is not a
    #: constant a default could hold.
    seed: Literal["today"] | None = None


@dataclass(frozen=True, kw_only=True)
class List(Widget):
    """A list of values typed by hand. The per-item control is derived by
    the host from the element type, so none is declared here."""

    kind: Literal["list"] = "list"
    min_items: int | None = None
    max_items: int | None = None


@dataclass(frozen=True, kw_only=True)
class SchemaBuilder(Widget):
    """A schema an author builds field by field — name, type, description."""

    kind: Literal["schema-builder"] = "schema-builder"
    schema: dict[str, Any] | None = None
    allow_additional: bool = True
    #: The field types the builder offers — the host's vocabulary, as data.
    field_types: tuple[Choice, ...] = ()


@dataclass(frozen=True, kw_only=True)
class CodeEditor(Widget):
    """Source a person writes, highlighted for ``language``."""

    kind: Literal["code-editor"] = "code-editor"
    language: str = "python"
    min_length: int | None = None
    max_length: int | None = None


@dataclass(frozen=True, kw_only=True)
class TemplateTextarea(Widget):
    """Text with placeholders. Each placeholder is an input."""

    kind: Literal["template-textarea"] = "template-textarea"
    rows: int = 4


@dataclass(frozen=True, kw_only=True)
class EntityDropdown(Widget):
    """Choices the host resolves — documents, say."""

    kind: Literal["entity-dropdown"] = "entity-dropdown"
    entity_kind: str = ""
    multiple: bool = False


@dataclass(frozen=True, kw_only=True)
class IfElseBuilder(Widget):
    """Conditions an author builds from the host's operators."""

    kind: Literal["if-else-builder"] = "if-else-builder"
    #: The operators the builder offers — the host's operator table, as data.
    operators: tuple[OperatorChoice, ...] = ()


@dataclass(frozen=True, kw_only=True)
class Tags(Widget):
    """Free-form labels a person adds one at a time."""

    kind: Literal["tags"] = "tags"


@dataclass(frozen=True, kw_only=True)
class TableInput(Widget):
    """A table an author types or pastes in, column types and all."""

    kind: Literal["table-input"] = "table-input"
    min_rows: int = 1
    min_columns: int = 1
    #: The column types an editor offers when a person corrects a guessed
    #: one — the host's scalar types, as data, each with the title its
    #: dtype declares.
    column_types: tuple[Choice, ...] = ()


#: The union of every widget, discriminated by ``kind``: the type of
#: ``Input.widget``, and what makes the JSON schema say which fields a
#: dropdown has and a number does not.
#:
#: Built once at import from the subclasses in this module, so a widget a
#: host declares elsewhere is not in the union and an ``Input`` carrying it
#: is refused by pydantic. That is deliberate: conductor ships the
#: controls and a host ships the vocabulary inside them as data. A new
#: control is a change here, since the component that renders each
#: ``kind`` has to exist in the host's frontend anyway.
AnyWidget = Annotated[
    Union[tuple(Widget.__subclasses__())],  # noqa: UP007
    Discriminator("kind"),
]
