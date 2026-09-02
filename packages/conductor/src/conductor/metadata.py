"""The records that say what fields a node has.

A node is its fields, and this file names them. ``Field`` is what the two
sides have in common; ``Output`` adds the one fact only a produced value has.

``Interface.of`` writes them, reading one ``run`` signature. Compile, the
engine and the editor read them, and nothing re-encodes them on the way out:
these records *are* the schema a palette publishes.
"""

from dataclasses import dataclass, field
from typing import Any

from conductor.dtype import DTypeRef
from conductor.types import WidgetType

__all__ = [
    "Field",
    "Output",
    "InputMetadata",
    "OutputMetadata",
]


@dataclass(frozen=True, kw_only=True)
class Field:
    """One named part of a node — an input or an output.

    A node consists of fields. This is what both sides have in common, and
    it is a real parent rather than four field names written twice: a
    ``Ref`` addresses one of these, ``Problem.field`` names one, and
    ``GraphNode.fields`` carries a title for each.

    Nobody constructs a bare ``Field``: what exists is an ``Input`` or an
    ``Output``, minted by ``Interface.of`` from a ``run`` signature and by a
    placement's roster hooks.

    ``dtype`` is a ``DTypeRef``: the declared type in Python, its wire form
    in JSON — a ``DType``'s ``describe()``, ``{"id": "any"}`` for the
    ``Any`` input, and ``null`` for the static type of an input
    no cable can reach, since nothing travels on it. That is what
    makes this record its own schema — a palette is these records
    dumped, and nothing re-encodes them.

    ``kw_only`` so a subclass can add a *required* field after the defaults
    here — ``Input.widget`` is required, and positional dataclass ordering
    would otherwise forbid it.
    """

    name: str
    dtype: DTypeRef
    title: str
    description: str | None = None


@dataclass(frozen=True, kw_only=True)
class Output(Field):
    """A field a value is produced on. Adds one fact to ``Field``: ``choice``.

    Named rather than left as a bare ``Field`` because the two sides read
    differently at every call site. A widget, a default and ``optional``
    are not here: they are not *blank* on an output, they cannot mean
    anything there, which is why this is a sibling of ``Input`` and
    not the same record with nullable fields.

    An output has a **handle** — the connection point an edge attaches to —
    and no widget. A widget is a control through which a person supplies a
    value, and nobody supplies a result.

    This is the *derived record*. What an author writes is a ``Result`` on
    the return type or on a record field, and ``outputs_of`` turns that
    declaration into these. From there the readers are compile — condition
    derivation reads ``choice``, the wires pass reads ``dtype`` — the editor,
    which draws one handle per output, and ``unpack``, which splits what
    ``run`` returned across them by name.

    No ``download`` and no ``filename``: a value is downloadable
    because of what it is, and a file carries its own name.
    """

    #: The group of exclusive alternatives this output belongs to:
    #: outputs of one placement sharing a value are alternatives, exactly
    #: one of which is produced — Hvis/ellers's branches. A contract fact,
    #: read by compile's condition derivation and by the editor («disse
    #: optræder aldrig sammen»), never by the engine, which propagates
    #: ``SKIPPED`` without knowing why. ``None`` means unconditional.
    choice: str | None = None


@dataclass(frozen=True)
class InputMetadata:
    """Pre-computed metadata for a single node input parameter."""

    name: str
    type_str: str
    label: str
    description: str | None = None
    widget: WidgetType = WidgetType.TEXT
    default: Any = None
    optional: bool = False
    expects_list: bool = False
    uses_connection_list: bool = False
    disable_handle: bool = False
    widget_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A ``list[...]`` input fans several upstream sources into a list. Derive
        # ``expects_list`` from the type so metadata built by hand carries the same
        # flag the registry derives at registration, rather than defaulting to
        # False and silently string-joining a fan-in. Only fills the default; an
        # explicit True is left untouched.
        if not self.expects_list and self.type_str.startswith("list["):
            object.__setattr__(self, "expects_list", True)


@dataclass(frozen=True)
class OutputMetadata:
    """Pre-computed metadata for a single node output."""

    name: str
    type_str: str
    label: str
    description: str | None = None
    optional: bool = False
    download: bool = False
    filename: str | None = None
