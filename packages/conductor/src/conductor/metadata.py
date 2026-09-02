"""The records that describe a node's fields.

``Field`` is what an input and an output have in common: name, type,
title, description. ``Output`` adds ``choice``. The input-side record,
which also carries a widget and a default, lands with the registry that
derives a node's interface from its ``run`` signature.

These records are their own schema: ``dtype`` is a ``DTypeRef``, so
dumping a record through pydantic gives the type's ``describe()``, and a
palette is simply these records dumped.
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

    ``name`` is what wires and bindings refer to (the ``field`` half of a
    ``Ref``); ``title`` and ``description`` are for a person. ``dtype`` is
    the declared type — a ``DType``, ``Any`` for an input that only routes
    a value, or a plain static type for an input no cable can reach, which
    serialises as ``null``.

    Nobody constructs a bare ``Field``; a node has outputs and inputs.
    Keyword-only so a subclass can add a required field after the defaults
    here.
    """

    name: str
    dtype: DTypeRef
    title: str
    description: str | None = None


@dataclass(frozen=True, kw_only=True)
class Output(Field):
    """A field a node produces a value on. Adds ``choice`` to ``Field``.

    Derived by ``outputs_of`` from the ``Result`` an author wrote on
    ``run``'s return type. It carries no widget, default or ``optional``:
    those are facts about how a person supplies a value, and nobody
    supplies a result. Nor ``download`` / ``filename``: whether a value can
    be downloaded follows from its type.
    """

    #: Outputs of one node that share a ``choice`` are exclusive alternatives:
    #: exactly one of them is produced per run (the two branches of an
    #: if/else node). ``None`` means the output is always produced. Read by
    #: the compiler and the editor, never by the engine, which only
    #: propagates the skip.
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
