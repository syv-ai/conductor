"""``Context[X]``: the run's context, one object the host passes for a run, which a node takes as its first parameter.

Example::

    @dataclass(frozen=True)
    class HostContext:
        caller: Caller
        http_client: HttpClient

    registry = NodeRegistry(context_type=HostContext)

    @registry.node("sign", title="Sign", description="Signs a text with the name of the person running it.")
    def sign(ctx: Context[HostContext], text: str) -> str:
        return f"{text}, {ctx.caller.name}"

    graph.run(text="Done", context=HostContext(caller=Caller("Ida"), http_client=client))

``ctx`` is not an input of the node: no edge reaches it, the graph record never mentions it, a run never saves it,
and it plays no part in whether a call's earlier outcome is reused. A unit test calls ``sign(host_context, "Done")``.

A node may name the registry's class, a base of it, or a ``Protocol`` the class satisfies. A node that does not know
the host's class, such as a pre-built one, names a ``Protocol`` with the members it reads::

    class HasHttpClient(Protocol):
        @property
        def http_client(self) -> HttpClient: ...

Write a member as a read-only property: a frozen dataclass's fields can't be set, so a type checker says it does
not satisfy a Protocol whose member is a plain attribute. Compiling a graph checks each node's ``Context[X]``
against the registry's class: ``issubclass`` for a class, ``what_the_context_lacks`` for a Protocol.
"""

from __future__ import annotations

import inspect
import typing
from dataclasses import dataclass
from typing import Annotated, Any, Protocol, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema

from conductor._internal.problem import NodeDeclarationError
from conductor._internal.problem import declarations as declaration

_T = TypeVar("_T")


class _Context:
    """The mark ``Context`` leaves on a parameter. ``@node`` takes it off a node's first parameter; anywhere else in
    a hint it refuses to be read."""

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Refuses a ``Context`` read as an input's or output's type: on a later or keyword-only parameter, inside
        ``| None`` or a list, or as the return type.

        Raises:
            NodeDeclarationError: Always (``context_misplaced``).
        """
        raise NodeDeclarationError(declaration.context_misplaced())

#: ``Context[X]``: marks a node's first parameter as the run's context, the object the host passes as ``context=``.
#: To a type checker the parameter is an ``X``.
Context = Annotated[_T, _Context()]


def is_context(hint: Any) -> bool:
    """Whether ``hint`` is ``Context[X]`` for some ``X``."""
    return typing.get_origin(hint) is Annotated and any(isinstance(mark, _Context) for mark in typing.get_args(hint)[1:])


@dataclass(frozen=True, slots=True)
class ContextParameter:
    """A node's first parameter when it takes the run's context. For ``sign(ctx: Context[HostContext], text: str)``,
    ``ContextParameter("ctx", HostContext)``.

    Attributes:
        name: The parameter's name. No input the node gets may have it.
        cls: The class inside ``Context[...]``: the registry's class, a base of it, or a ``Protocol``.
    """

    name: str
    cls: type


def is_protocol(cls: type) -> bool:
    """Whether ``cls`` is a ``Protocol``: a class that lists ``Protocol`` among its own bases, as every Protocol must."""
    return Protocol in cls.__bases__


def what_the_context_lacks(context_type: type, protocol: type) -> list[str]:
    """The members of ``protocol`` that a context of class ``context_type`` lacks; ``[]`` when it has them all.

    Example, with ``HostContext`` above: ``HasHttpClient`` gives ``[]``. A Protocol with a member ``database`` gives
    ``["database"]``, and one whose ``http_client`` is a ``str`` gives ``["http_client as a str"]``.

    A member is compared by its name, and by its class where both sides declare a plain class (an attribute's
    annotation, a property's return type). A method is compared by its name only.
    """
    members: set[str] = getattr(protocol, "__protocol_attrs__")  # typing sets it on every Protocol; get_protocol_members reads it from 3.13
    lacks: list[str] = []
    for member in sorted(members):
        if not hasattr(context_type, member) and member not in typing.get_type_hints(context_type):
            lacks.append(member)
            continue
        wanted, declared = _declared_class(protocol, member), _declared_class(context_type, member)
        if wanted is not None and declared is not None and not issubclass(declared, wanted):
            lacks.append(f"{member} as a {wanted.__name__}")
    return lacks


def _declared_class(cls: type, member: str) -> type | None:
    """The plain class ``cls`` declares ``member`` as: its annotation, or the return type of a property; ``None`` for
    anything else (a method, a union, a generic)."""
    declared = typing.get_type_hints(cls).get(member)
    if declared is None and isinstance(found := inspect.getattr_static(cls, member, None), property) and found.fget is not None:
        declared = typing.get_type_hints(found.fget).get("return")
    return declared if isinstance(declared, type) else None
