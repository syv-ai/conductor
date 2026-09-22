"""``Interface`` — what a node's ``run`` signature declares, read once.

A node declares its inputs and outputs by annotating ``run``::

    def run(
        self,
        text: Annotated[Text, Param(title="Text", widget=Textarea())],
        language: Annotated[Text, Param(title="Language", widget=Dropdown(choices=...))] = Text("en"),
    ) -> Annotated[Text, Result(title="Translation")]:
        ...

``Interface.of(run)`` walks that signature once and produces the input
records, the output records and what the caller must provide.
``model_of`` builds the pydantic model that validates a call against a
tuple of inputs. Nothing else reads the signature, so there is one place
the two could disagree, and it is here.

A parameter carries at most one ``Param``; a parameter with none is
``Param()``, titled by its name, with no widget. A widget written bare in
``Annotated`` is refused: it belongs on the ``Param``. Three things the
walk understands beyond ``Annotated[DType, Param(...)]``:

* ``Any`` in place of a ``DType`` — the input accepts whatever is connected to
  it. The type that actually arrives is recorded when the graph is
  compiled, and the node's ``compute_outputs`` types its outputs from it.
* ``**inputs: Single`` (or ``**inputs: Series``) — an open interface: every
  name connected to the node becomes an input, received as one value (or, for
  ``Series``, as a whole series). The interface records only that it
  is open and in which shape; the inputs themselves are made from
  the edges when the graph is compiled.
* A parameter whose ``Param`` says ``show_handle=False`` cannot be connected, so
  it may declare any pydantic-validatable type (a schema, a list of
  branches). A ``DType`` is required exactly where an edge can land.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Annotated, Any, Callable, Literal, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model

from conductor.dtype import DType, Single, dtype_of
from conductor.metadata import Input, Output, Param, Result
from conductor.returns import outputs_of
from conductor.series import Series
from conductor.widgets import Widget


@dataclass(frozen=True)
class FromRun:
    """Marks a parameter that the caller of the graph supplies, not the graph.

    ``clock: Annotated[Clock, FromRun()]`` says: this is not an
    input — no widget, no handle, no binding — but a value the host hands
    to ``execute(from_run={Clock: ...})``, which the engine passes
    in by type. ``Interface.of`` collects such parameters into
    ``Interface.needs``, and ``execute`` refuses to start a graph whose
    nodes need a type it was not given.

    Not a context bag: a node receives only what its own signature names.
    """


@dataclass(frozen=True)
class Interface:
    """A node's inputs, outputs and needs.

    Three things carry one. What one version declares: ``Interface.of``
    reads it off the ``run`` signature when a node class is defined. What
    one placed node actually has once the compiler has asked its hooks and
    typed its edges: ``CompiledGraph.node(node_id).interface``, the same record with
    ``returns``, ``needs`` and ``open`` copied from the version. What a
    whole graph takes and returns: ``CompiledGraph.interface``, with
    inputs named by address and wearing their nodes' titles. Derived,
    never written by hand; frozen, so the derivation is the only writer.

    Not stored, and not shared between versions: a version is a
    signature, so two versions are two interfaces.
    """

    inputs: tuple[Input, ...]
    outputs: tuple[Output, ...]
    #: The declared return type with ``Annotated`` stripped — a ``DType`` (or
    #: ``Any``), a record class, or ``Mapping`` — which is what ``unpack``
    #: uses to split ``run``'s return across the outputs.
    returns: Any
    #: The types the caller must provide, by parameter name, from every
    #: ``FromRun`` parameter. ``execute`` refuses to start a graph whose
    #: nodes need a type it was not given.
    needs: dict[str, type] = field(default_factory=dict)
    #: The shape of an open interface, or ``None`` for a closed one:
    #: ``"single"`` for ``**inputs: Single`` (each connected name received as
    #: one value), ``"series"`` for ``**inputs: Series`` (each received as a
    #: whole series). The inputs themselves are made from the edges when
    #: the graph is compiled, so only the shape is recorded here.
    open: Literal["single", "series"] | None = None

    @classmethod
    def of(cls, func: Callable[..., Any]) -> "Interface":
        """Read ``func``'s signature into an ``Interface``. A leading ``self`` is skipped."""
        signature = inspect.signature(func)
        hints = get_type_hints(func, include_extras=True)
        inputs, needs, open_shape = cls._extract_inputs(signature, hints)
        returns, outputs = cls._declared_outputs(hints)
        taken = {i.name for i in inputs} & {o.name for o in outputs}
        if taken:
            raise TypeError(
                f"{sorted(taken)} named on both sides: a field name is unique within a node, "
                "because a Ref (node_id, field) is an address on either side"
            )
        return cls(inputs=inputs, outputs=outputs, returns=returns, needs=needs, open=open_shape)

    @staticmethod
    def _extract_inputs(
        signature: inspect.Signature, hints: dict[str, Any]
    ) -> tuple[tuple[Input, ...], dict[str, type], Literal["single", "series"] | None]:
        """One walk over the parameters: the ``Input`` records, the ``FromRun``
        needs by parameter name, and the shape of an open interface (``"single"``,
        ``"series"`` or ``None``)."""
        inputs: list[Input] = []
        needs: dict[str, type] = {}
        open_shape: Literal["single", "series"] | None = None
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            annotation = hints.get(name, parameter.annotation)
            if parameter.kind not in (inspect.Parameter.VAR_KEYWORD, inspect.Parameter.VAR_POSITIONAL):
                _refuse_name(name)
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                if annotation is Single:
                    # ``**inputs: Single``: an open interface, every connected name
                    # received as one value. Only the shape is recorded; the
                    # inputs are made from the edges at compile time.
                    open_shape = "single"
                elif annotation is Series:
                    # ``**inputs: Series``: an open interface, every edge a
                    # reduction.
                    open_shape = "series"
                elif isinstance(annotation, type) and issubclass(annotation, Series):
                    raise TypeError(
                        f"**{name}: {annotation.__name__} would close the interface silently; an open "
                        f"interface whose every edge is a reduction is **{name}: Series"
                    )
                # Any other ``**values``: the inputs this node's ``compute_inputs``
                # adds arrive here by name. The hook declares them, not the signature.
                continue
            if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
                raise TypeError(
                    f"parameter *{name}: a node's inputs are named, so an edge or a binding "
                    "can land on each; *args names none"
                )
            if annotation is Single or _declared(annotation) is Single:
                raise TypeError(
                    f"parameter {name!r}: Single is spelled on **inputs only; "
                    "a named parameter declares a DType, or Any for whatever arrives"
                )
            if _annotation_of(annotation, FromRun) is not None:
                needs[name] = get_args(annotation)[0]
                continue
            if _annotation_of(annotation, Widget) is not None:
                raise TypeError(
                    f"parameter {name!r} carries a bare widget; the widget goes on the Param — "
                    "Annotated[Text, Param(title=..., widget=Textarea())]"
                )
            if Result.on(annotation) is not None:
                raise TypeError(
                    f"parameter {name!r} carries a Result; a parameter carries a Param, and Result belongs on the return"
                )
            param = Param.on(annotation) or Param()
            dtype = dtype_of(annotation)
            if isinstance(dtype, type) and issubclass(dtype, Series) and dtype.element is None:
                raise TypeError(f"parameter {name!r} is a bare Series; declare Series[X] with the element type")
            if param.show_handle:
                # An edge can land here, so the type must be one an edge carries:
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
            has_default = parameter.default is not inspect.Parameter.empty
            inputs.append(
                Input(
                    name=name,
                    dtype=dtype,
                    title=param.title or name,
                    description=param.description,
                    widget=param.widget,
                    show_handle=param.show_handle,
                    default=parameter.default if has_default else None,
                    optional=has_default,
                )
            )
        return tuple(inputs), needs, open_shape

    @staticmethod
    def _declared_outputs(hints: dict[str, Any]) -> tuple[Any, tuple[Output, ...]]:
        """The declared return type and the outputs it declares.

        A ``run`` with no return annotation is an error, not a node with no
        outputs: the engine would have nowhere to put what it returns. A
        ``Mapping`` return declares no outputs here; ``compute_outputs``
        supplies them when the graph is compiled.
        """
        if "return" not in hints:
            raise TypeError("run() must declare a return type")
        return outputs_of(hints["return"])


def model_of(inputs: tuple[Input, ...]) -> type[BaseModel]:
    """The pydantic model that validates a call against ``inputs``.

    A function rather than a field on ``Interface``: the inputs a call is
    validated against are usually a placed node's own interface rather than the bare
    declaration, and a model class held on a frozen record would make two
    interfaces derived from one signature compare unequal. An ``Any`` input
    validates anything; the compiler has already established what arrives
    there.

    A field is the input's own name, except where ``BaseModel`` already
    has that name (``schema``, ``json``, ``model_config``): that input gets
    the field ``schema_`` — its name with underscores added until no
    ``BaseModel`` attribute and no other input has it — and is taken by its
    name as the alias, so it neither shadows the model nor collides with an
    input really named ``schema_``. Read the values back by alias where there
    is one (``CompiledNode.validate`` does).
    """
    taken = {inp.name for inp in inputs}
    fields: dict[str, Any] = {}
    for inp in inputs:
        field_name = inp.name
        while hasattr(BaseModel, field_name) or (field_name != inp.name and field_name in taken):
            field_name += "_"
        taken.add(field_name)
        alias = None if field_name == inp.name else inp.name
        fields[field_name] = (inp.dtype, Field(inp.default if inp.optional else ..., alias=alias))
    return create_model(
        "Inputs",
        __config__=ConfigDict(extra="ignore", arbitrary_types_allowed=True),
        **fields,
    )


def _refuse_name(name: str) -> None:
    """Refuse a parameter name no input can carry: one starting with an
    underscore, private by convention and never a field an author fills."""
    if name.startswith("_"):
        raise TypeError(f"parameter {name!r}: an input's name cannot start with an underscore")


def _declared(hint: Any) -> Any:
    """The type under ``Annotated[...]``, or the hint itself."""
    return get_args(hint)[0] if get_origin(hint) is Annotated else hint


def _annotation_of(hint: Any, kind: type) -> Any:
    """The first ``Annotated`` extra of type ``kind``, or ``None``."""
    if get_origin(hint) is not Annotated:
        return None
    for extra in get_args(hint)[1:]:
        if isinstance(extra, kind):
            return extra
    return None
