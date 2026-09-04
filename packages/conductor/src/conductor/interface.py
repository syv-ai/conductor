"""``Interface`` — what a node's ``run`` signature declares, read once.

A node declares its inputs and outputs by annotating ``run``::

    def run(
        self,
        text: Annotated[Text, Textarea(title="Text")],
        language: Annotated[Text, Dropdown(title="Language", choices=...)] = Text("en"),
    ) -> Annotated[Text, Result(title="Translation")]:
        ...

``Interface.of(run)`` walks that signature once and produces the input
records, the output records and what the caller must provide.
``model_of`` builds the pydantic model that validates a call against a
tuple of inputs. Nothing else reads the signature, so there is one place
the two could disagree, and it is here.

Three things the walk understands beyond ``Annotated[DType, Widget]``:

* ``Any`` in place of a ``DType`` — the input accepts whatever is wired to
  it. The type that actually arrives is recorded when the flow is
  compiled, and the node's ``compute_outputs`` types its outputs from it.
* ``**inputs: Single`` (or ``**inputs: Series``) — an open roster: every
  name wired to the node becomes an input, received as one value (or, for
  ``Series``, as a whole series). The interface records only that the
  roster is open and in which shape; the inputs themselves are made from
  the wiring when the flow is compiled.
* A parameter whose widget says ``show_handle=False`` cannot be wired, so
  it may declare any pydantic-validatable type (a schema, a list of
  branches). A ``DType`` is required exactly where a cable can land.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Literal, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model

from conductor.dtype import DType, Single, dtype_of
from conductor.metadata import Input, Output
from conductor.returns import outputs_of
from conductor.series import Series
from conductor.widgets import Widget


@dataclass(frozen=True)
class Provided:
    """Marks a parameter that the caller of the flow supplies, not the flow.

    ``identity: Annotated[RunnerIdentity, Provided()]`` says: this is not an
    input — no widget, no handle, no binding — but a value the host hands
    to ``execute(provides={RunnerIdentity: ...})``, which the engine passes
    in by type. ``Interface.of`` collects such parameters into
    ``Interface.needs``, and ``execute`` refuses to start a flow whose
    nodes need a type it was not given.

    Not a context bag: a node receives only what its own signature names.
    """


@dataclass(frozen=True)
class Interface:
    """What one version of a node declares: its inputs, outputs and needs.

    Derived, never written by hand. ``Interface.of`` builds one from a
    ``run`` signature when a node class is defined, and the compiler builds
    one for a whole flow, with inputs named by address and wearing their
    placements' titles. Frozen, so the derivation is the only writer.

    Not what a particular *placement* of the node ends up with — a
    placement's roster may be reshaped by the values it holds. Not stored,
    and not shared between versions: a version is a signature, so two
    versions are two interfaces.
    """

    inputs: tuple[Input, ...]
    outputs: tuple[Output, ...]
    #: The declared return type with ``Annotated`` stripped — a ``DType`` (or
    #: ``Any``), a record class, or ``Mapping`` — which is what ``unpack``
    #: uses to split ``run``'s return across the outputs.
    returns: Any
    #: The types the caller must provide, by parameter name, from every
    #: ``Provided`` parameter. ``execute`` refuses to start a flow whose
    #: nodes need a type it was not given.
    needs: dict[str, type] = field(default_factory=dict)
    #: The shape of an open roster, or ``None`` for a closed one:
    #: ``"single"`` for ``**inputs: Single`` (each wired name received as
    #: one value), ``"series"`` for ``**inputs: Series`` (each received as a
    #: whole series). The inputs themselves are made from the wiring when
    #: the flow is compiled, so only the shape is recorded here.
    open: Literal["single", "series"] | None = None

    @classmethod
    def of(cls, func: Callable[..., Any]) -> "Interface":
        """Read ``func``'s signature into an ``Interface``. A leading ``self`` is skipped."""
        signature = inspect.signature(func)
        hints = get_type_hints(func, include_extras=True)
        inputs, needs, open_roster = _extract_inputs(signature, hints)
        returns, outputs = _extract_outputs(hints)
        taken = {i.name for i in inputs} & {o.name for o in outputs}
        if taken:
            raise TypeError(
                f"{sorted(taken)} named on both sides: a field name is unique within a node, "
                "because a Ref (node_id, field) is an address on either side"
            )
        return cls(inputs=inputs, outputs=outputs, returns=returns, needs=needs, open=open_roster)


def model_of(inputs: tuple[Input, ...]) -> type[BaseModel]:
    """The pydantic model that validates a call against ``inputs``.

    A function rather than a field on ``Interface``: the inputs a call is
    validated against are usually a placement's roster rather than the bare
    declaration, and a model class held on a frozen record would make two
    interfaces derived from one signature compare unequal. An ``Any`` input
    validates anything; the compiler has already established what arrives
    there.
    """
    return create_model(
        "Inputs",
        __config__=ConfigDict(extra="ignore", arbitrary_types_allowed=True),
        **{
            inp.name: (inp.dtype, inp.default if inp.optional else ...)
            for inp in inputs
        },
    )


def _extract_inputs(
    signature: inspect.Signature, hints: dict[str, Any]
) -> tuple[tuple[Input, ...], dict[str, type], Literal["single", "series"] | None]:
    """One walk over the parameters: the ``Input`` records, the ``Provided``
    needs by parameter name, and the open roster's shape (``"single"``,
    ``"series"`` or ``None``)."""
    inputs: list[Input] = []
    needs: dict[str, type] = {}
    open_roster: Literal["single", "series"] | None = None
    for name, param in signature.parameters.items():
        if name == "self":
            continue
        annotation = hints.get(name, param.annotation)
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            if annotation is Single:
                # ``**inputs: Single``: an open roster, every wired name
                # received as one value. Only the shape is recorded; the
                # inputs are made from the wiring at compile time.
                open_roster = "single"
            elif annotation is Series:
                # ``**inputs: Series``: an open roster, every wire a
                # reduction.
                open_roster = "series"
            # Any other ``**values``: the inputs this node's ``compute_inputs``
            # adds arrive here by name. The hook declares them, not the signature.
            continue
        if annotation is Single or _declared(annotation) is Single:
            raise TypeError(
                f"parameter {name!r}: Single is spelled on **inputs only; "
                "a named parameter declares a DType, or Any for whatever arrives"
            )
        if _annotation_of(annotation, Provided) is not None:
            needs[name] = get_args(annotation)[0]
            continue
        widget = _annotation_of(annotation, Widget)
        if widget is None:
            raise TypeError(
                f"parameter {name!r} declares no widget — annotate it, "
                "e.g. Annotated[Text, Textarea(title=...)]"
            )
        dtype = dtype_of(annotation)
        if widget.show_handle:
            # A cable can land here, so the type must be one a wire carries:
            # a DType, or Any for "whatever arrives".
            if dtype is None:
                raise TypeError(
                    f"parameter {name!r} has a handle and must declare a DType or Any — "
                    f"got {annotation!r}"
                )
            if dtype is DType:
                raise TypeError(
                    f"parameter {name!r} must declare a concrete DType, or "
                    "Any for whatever arrives; the base would accept anything"
                )
        elif dtype is None:
            # No handle, so nothing travels: the declared type is a static
            # type, used to validate the value a person typed.
            dtype = _declared(annotation)
        has_default = param.default is not inspect.Parameter.empty
        # `title`, `description` and `show_handle` are copied off the widget
        # annotation onto the Input, like `dtype` is read off the annotation.
        # Nothing downstream reads them from the widget.
        inputs.append(
            Input(
                name=name,
                dtype=dtype,
                title=widget.title,
                description=widget.description,
                widget=widget,
                show_handle=widget.show_handle,
                default=param.default if has_default else None,
                optional=has_default,
            )
        )
    return tuple(inputs), needs, open_roster


def _declared(hint: Any) -> Any:
    """The type under ``Annotated[...]``, or the hint itself."""
    return get_args(hint)[0] if get_origin(hint) is Annotated else hint


def _extract_outputs(hints: dict[str, Any]) -> tuple[Any, tuple[Output, ...]]:
    """The declared return type and the outputs it declares.

    A ``run`` with no return annotation is an error, not a node with no
    outputs: the engine would have nowhere to put what it returns. A
    ``Mapping`` return declares no outputs here; the placement's computed
    roster supplies them.
    """
    if "return" not in hints:
        raise TypeError("run() must declare a return type")
    return outputs_of(hints["return"])


def _annotation_of(hint: Any, kind: type) -> Any:
    """The first ``Annotated`` extra of type ``kind``, or ``None``."""
    if get_origin(hint) is not Annotated:
        return None
    for extra in get_args(hint)[1:]:
        if isinstance(extra, kind):
            return extra
    return None
