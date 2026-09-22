"""The records that describe a node's fields — what an author writes, and what the node carries.

An author says what a person should read about a field inside
``Annotated`` on ``run``: ``Param`` on a parameter, ``Result`` on the
return. Both are a ``Field`` — a title and a description — and each adds
what only its side has. A ``Param`` says how the value gets there: the
``widget`` a person edits it with, if any, and whether an edge can reach
it. A ``Result`` says which ``choice`` group the output belongs to::

    def run(
        self,
        text: Annotated[Text, Param(title="Text", widget=Textarea())],
        rows: Annotated[Series[Text], Param(title="Rows")],
    ) -> Annotated[Text, Result(title="Summary")]:
        ...

The node ends up with an ``Input`` per parameter and an ``Output`` per
declared result: the author's record plus the ``name`` edges and bindings
refer to and the declared ``dtype``. ``Interface.of`` writes them when a
node class is defined; the compiler, the engine and an editor read them.
A ``Param`` with a widget is edited in place and reachable by an edge; a
``Param`` with no widget is filled by an edge only; ``show_handle=False``
closes the input to edges, so its type may be any static type. A
parameter with no ``Param`` at all is ``Param()``: titled by its name,
no widget, reachable.

These records are their own schema: ``dtype`` is a ``DTypeRef``, so a
dump gives the type's ``describe()``, and a palette is these records
dumped. A dump does not read back into one of these records: ``dtype``
takes a ``DType`` class, ``Any`` or a type, never a description, and the
class itself lives on the signature.
"""

from __future__ import annotations

from typing import Annotated, Any, get_args, get_origin

from conductor.dtype_ref import DTypeRef
from conductor.model import ConductorModel
from conductor.widgets import AnyWidget


class Field(ConductorModel):
    """What a person reads about one part of a node: a ``title`` and a ``description``.

    The half an input and an output share. ``title`` is optional on the
    author's records, ``Param`` and ``Result``, where it falls back to the
    parameter's or the output's name; the node's records, ``Input`` and
    ``Output``, always carry one. Nobody writes a bare ``Field``.
    """

    title: str | None = None
    description: str | None = None

    @classmethod
    def on(cls, hint: Any) -> Any:
        """The record of this class written inside ``Annotated`` on ``hint``, or ``None``."""
        if get_origin(hint) is not Annotated:
            return None
        return next((extra for extra in get_args(hint)[1:] if isinstance(extra, cls)), None)


class Param(Field):
    """What the author says about an input, inside ``Annotated`` on a ``run`` parameter.

    Adds to ``Field`` how the value gets there. ``widget`` is the control a
    person edits it with, or ``None`` for an input only an edge fills (a
    ``Series[X]`` gathered from other nodes, an ``Any`` passed through).
    ``show_handle`` is whether an edge can reach the input; ``False``
    closes it, so the parameter may declare a static type nothing travels
    on (a schema an author builds). Its twin on the return is ``Result``.
    ``Interface.of`` reads it once into an ``Input``; the ``Param`` itself
    is not kept.
    """

    widget: AnyWidget | None = None
    show_handle: bool = True


class Result(Field):
    """What the author says about an output, inside ``Annotated`` on ``run``'s return or on a field of the returned record.

    Adds ``choice`` to ``Field``: outputs of one node that share a
    ``choice`` are exclusive alternatives, exactly one of them produced per
    run, as with the two branches of an if/else node::

        if_true: Annotated[Text, Result(title="If true", choice="branches")]
        if_false: Annotated[Text, Result(title="If false", choice="branches")]

    Its twin on a parameter is ``Param``. ``outputs_of`` reads it once
    into an ``Output``; the ``Result`` itself is not kept.
    """

    choice: str | None = None

    def output(self, name: str, dtype: Any) -> Output:
        """The ``Output`` this declares for the field ``name`` of type ``dtype``."""
        return Output(name=name, dtype=dtype, title=self.title or name, description=self.description, choice=self.choice)


class Input(Param):
    """One parameter of ``run``, as the node carries it.

    A ``Param`` plus the ``name`` edges and bindings refer to (the ``field``
    half of a ``Ref``), the declared ``dtype``, and the parameter's
    ``default``. ``dtype`` is a ``DType`` or ``Any`` where the input has a
    handle, and any pydantic-validatable type where it has none (it
    serialises as ``null``). ``widget`` is typed as the union of every
    widget so the record's JSON schema is discriminated per control.

    ``Interface.of`` builds one per parameter; ``compute_inputs`` may build
    more for a node whose fields depend on its values; ``Asks`` carries
    them as the questions a person answers. Read by ``model_of`` to
    validate a call, by the compiler to type a typed-in value and to decide
    whether an edge may land, and by an editor to draw the row.
    """

    name: str
    dtype: DTypeRef
    title: str

    #: What the value is when nothing binds the input.
    default: Any = None

    #: Whether the parameter has a default at all. Kept separately because
    #: ``None`` is a legitimate default value.
    optional: bool = False


class Output(Result):
    """One declared output of ``run``, as the node carries it.

    A ``Result`` plus the ``name`` other nodes edge to and the declared
    ``dtype``. Derived by ``outputs_of`` from the return annotation. It
    carries no widget, default or ``optional``: those are facts about how
    a person supplies a value, and nobody supplies a result. ``choice`` is
    read by the compiler and the editor, never by the engine, which only
    propagates the skip.
    """

    name: str
    dtype: DTypeRef
    title: str

    def question(self) -> Input:
        """The question a person answers in this output's place: the same
        name, type, title and description, with no widget — the host picks
        the control from the type. What ``Asks()`` asks by default."""
        return Input(name=self.name, dtype=self.dtype, title=self.title, description=self.description)
