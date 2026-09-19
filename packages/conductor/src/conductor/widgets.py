"""Widgets — how a person edits an input.

A widget is the control an input is edited with, plus what that control
needs: a dropdown's choices, a number's range. Every widget is a frozen,
keyword-only record with a ``kind`` discriminator, and ``AnyWidget`` is the
union of all of them, so pydantic dumps a widget and publishes a JSON
schema per kind. An author writes one on the ``Param`` inside
``Annotated`` on a ``run`` parameter::

    language: Annotated[Text, Param(title="Language", widget=Dropdown(choices=(Choice(id="en", title="English"),)))]

A widget is the control and nothing more. What a person reads about the
input — its title and description — and whether an edge can reach it are
the ``Param``'s, not the widget's, so each fact is written once and a
dumped widget reads back into one. An input with no widget is filled by an
edge only: ``Param(title=...)`` alone declares it. Nor does a widget
change how the engine runs: a pause is a node that returns ``Asks``, not a
widget kind.

Conductor ships no default widget for any type: a text may be a
textarea, a single line or a dropdown, so every input declares its own.
The three controls whose plain names collide with a host's types are
named for what they are — ``TextWidget``, ``NumberWidget``,
``ListWidget`` — so ``from conductor.widgets import TextWidget`` sits
beside a host's ``Text`` without an alias.
"""

from __future__ import annotations

from abc import ABC
from typing import Annotated, Any, Literal, Union

from pydantic import ConfigDict, Discriminator, Field

from conductor.model import ConductorModel


class Widget(ConductorModel, ABC):
    """The base of every control: a ``kind`` and what that control needs.

    Never subclassed outside this module: ``AnyWidget`` is built from the
    subclasses declared here, and a control the host's frontend cannot
    render is not a control.
    """


class Choice(ConductorModel):
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


class OperatorChoice(ConductorModel):
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


class TextWidget(Widget):
    """Single-line text."""

    kind: Literal["text"] = "text"
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None


class Textarea(Widget):
    """Multi-line text."""

    kind: Literal["textarea"] = "textarea"
    min_length: int | None = None
    max_length: int | None = None
    rows: int = 4


class Dropdown(Widget):
    """Pick one of a declared vocabulary of ``Choice``s."""

    kind: Literal["dropdown"] = "dropdown"
    choices: tuple[Choice, ...] = ()


class Range(Widget):
    """A number picked on a slider, between declared bounds."""

    kind: Literal["range"] = "range"
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None


class FileUpload(Widget):
    """Files a person uploads. ``multiple`` births a series of them."""

    kind: Literal["file"] = "file"
    accept: tuple[str, ...] | None = None
    max_size_mb: float | None = None
    multiple: bool = False


class NumberWidget(Widget):
    """A number typed in, optionally bounded and optionally whole."""

    kind: Literal["number"] = "number"
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    integer_only: bool = False


class Switch(Widget):
    """A boolean, on or off."""

    kind: Literal["switch"] = "switch"


class DatePicker(Widget):
    """A date picked from a calendar."""

    kind: Literal["datepicker"] = "datepicker"
    min_date: str | None = None
    max_date: str | None = None
    #: ``"today"`` asks the editor to write today's date into the field
    #: when the node is placed. Declared here because "today" is not a
    #: constant a default could hold.
    seed: Literal["today"] | None = None


class ListWidget(Widget):
    """A list of values typed by hand. The per-item control is derived by
    the host from the element type, so none is declared here."""

    kind: Literal["list"] = "list"
    min_items: int | None = None
    max_items: int | None = None


class SchemaBuilder(Widget):
    """A schema an author builds field by field — name, type, description."""

    #: ``schema`` on the wire and as a keyword; ``schema_`` as an attribute,
    #: because ``BaseModel.schema`` is taken: ``builder.schema`` is pydantic's
    #: deprecated classmethod, not the value. A dump says ``schema`` unless
    #: asked for ``by_alias=False``.
    model_config = ConfigDict(validate_by_name=True, validate_by_alias=True, serialize_by_alias=True)

    kind: Literal["schema-builder"] = "schema-builder"
    schema_: Annotated[dict[str, Any] | None, Field(alias="schema")] = None
    allow_additional: bool = True
    #: The field types the builder offers — the host's vocabulary, as data.
    field_types: tuple[Choice, ...] = ()


class CodeEditor(Widget):
    """Source a person writes, highlighted for ``language``."""

    kind: Literal["code-editor"] = "code-editor"
    language: str = "python"
    min_length: int | None = None
    max_length: int | None = None


class TemplateTextarea(Widget):
    """Text with placeholders. Each placeholder is an input."""

    kind: Literal["template-textarea"] = "template-textarea"
    rows: int = 4


class EntityDropdown(Widget):
    """Choices the host resolves — documents, say."""

    kind: Literal["entity-dropdown"] = "entity-dropdown"
    entity_kind: str = ""
    multiple: bool = False


class IfElseBuilder(Widget):
    """Conditions an author builds from the host's operators."""

    kind: Literal["if-else-builder"] = "if-else-builder"
    #: The operators the builder offers — the host's operator table, as data.
    operators: tuple[OperatorChoice, ...] = ()


class Tags(Widget):
    """Free-form labels a person adds one at a time."""

    kind: Literal["tags"] = "tags"


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
