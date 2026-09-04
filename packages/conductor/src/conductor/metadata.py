"""The records that describe a node's fields.

``Field`` is what an input and an output have in common: name, type,
title, description. ``Input`` adds how a person supplies the value (widget,
default, optionality, whether a cable can reach it); ``Output`` adds
``choice``. The registry writes them when it derives a node's interface
from its ``run`` signature; the compiler, the engine and the editor read them.

These records are their own schema: ``dtype`` is a ``DTypeRef``, so
dumping a record through pydantic gives the type's ``describe()``, and a
palette is simply these records dumped.
"""

from dataclasses import dataclass
from typing import Any

from conductor.dtype_ref import DTypeRef
from conductor.widgets import AnyWidget


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


@dataclass(frozen=True, kw_only=True)
class Input(Field):
    """A field a value is supplied to — one parameter of ``run``.

    Adds to ``Field`` how the value gets there: the ``widget`` a person
    edits it with, whether a cable can reach it (``show_handle``), and the
    parameter's ``default``. ``Interface.of`` builds one per ``run``
    parameter, copying ``title``, ``description`` and ``show_handle`` off
    the widget annotation onto the record; ``compute_inputs`` may build
    more for a placement whose fields depend on its values::

        Input(name="text", dtype=Text, title="Text", widget=Textarea(title="Text"))

    ``dtype`` is a ``DType`` or ``Any`` where the input has a handle, and
    any pydantic-validatable type where it has none (a schema an author
    fills in; it serialises as ``null``). ``widget`` is typed as the union
    of every widget so the record's JSON schema is discriminated per
    control. Read by ``model_of`` to validate a call, by the compiler to
    type a typed-in value and to decide whether a cable may land, and by
    an editor to draw the row.
    """

    #: How a person supplies the value. Required: an input with no widget is
    #: a broken declaration, not one that falls back to a default control.
    widget: AnyWidget

    #: Whether a cable can reach this input. A fact about the field, copied
    #: off the widget annotation where the author wrote it.
    show_handle: bool = True

    #: What the value is when nothing binds the input.
    default: Any = None

    #: Whether the parameter has a default at all. Kept separately because
    #: ``None`` is a legitimate default value.
    optional: bool = False
