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
against the registry's class: for a class, that the registry's class is it or a subclass of it (``issubclass``); for a
Protocol, that the class has every member the Protocol declares, by name (``ContextParameter.fit``). It does not
check the members' types: an ``http_client: str`` has the name ``HasHttpClient`` asks for. Whether the types fit is
the host's type checker's job.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from typing import Annotated, Any, TypeVar

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema
from typing_extensions import get_protocol_members, is_protocol

from conductor._internal.problem import (
    NodeDeclarationError,
    Problem,
    compile_problems,
    declaration_problems,
)

_T = TypeVar("_T")


class _Context:
    """The metadata ``Context[X]`` puts beside ``X``: ``Context[X]`` is ``Annotated[X, _Context()]``. ``@node`` takes
    it off a node's first parameter; anywhere else in a hint it refuses to be read."""

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Refuses a ``Context`` read as an input's or output's type: on a later or keyword-only parameter, inside
        ``| None`` or a list, or as the return type.

        Raises:
            NodeDeclarationError: Always (``context_misplaced``).
        """
        raise NodeDeclarationError(declaration_problems.context_misplaced())

#: ``Context[X]``: the type of a node's first parameter that is the run's context, the object the host passes as
#: ``context=``. To a type checker the parameter is an ``X``.
Context = Annotated[_T, _Context()]


def is_context(hint: Any) -> bool:
    """Whether ``hint`` is ``Context[X]`` for some ``X``."""
    return typing.get_origin(hint) is Annotated and any(isinstance(metadata, _Context) for metadata in typing.get_args(hint)[1:])


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


    def fit(self, node_id: str, context_type: type | None) -> Problem | None:
        """Says whether the class a registry names as its context, ``context_type``, is what this parameter asks for.

        Example, with ``HostContext`` above: ``sign(ctx: Context[HostContext], text: str)`` fits ``HostContext`` and
        any subclass of it. ``fetch(ctx: Context[HasHttpClient], url: str)`` fits a class with an ``http_client``,
        of any type. On a registry that names no class, neither fits.

        Args:
            node_id: The node whose first parameter this is, for the problem's ``loc``.
            context_type: The registry's ``context_type``; ``None`` when it names none.

        Returns:
            ``None`` when it fits; else ``context_does_not_fit`` at ``(node_id, name)``: the class is not ``cls`` or a
            subclass of it, it lacks members of the Protocol ``cls`` (variant ``lacks``), or the registry names no
            class (variant ``no_context_type``).
        """
        needs = self.cls.__name__
        if context_type is None:
            return compile_problems.context_does_not_fit_no_context_type(node_id, self.name, needs=needs)
        if not is_protocol(self.cls):
            return None if issubclass(context_type, self.cls) else compile_problems.context_does_not_fit(node_id, self.name, needs=needs, context_type=context_type.__name__)
        lacks = _lacks(context_type, self.cls)
        return compile_problems.context_does_not_fit_lacks(node_id, self.name, needs=needs, context_type=context_type.__name__, lacks=lacks) if lacks else None


def _lacks(context_type: type, protocol: type) -> list[str]:
    """The names of the members of ``protocol`` that the class ``context_type`` does not have; ``[]`` when it has them all.

    Example, with ``HostContext`` above: ``HasHttpClient`` gives ``[]``, and a Protocol with a member ``database``
    gives ``["database"]``. The class has a member when it has an attribute, method or property of that name, or
    annotates a field of that name (a dataclass field with no default). The member's type is not compared: a class
    with ``http_client: str`` has every member ``HasHttpClient`` declares. Whether the types fit is the host's type
    checker's job.
    """
    annotated = typing.get_type_hints(context_type)
    return [member for member in sorted(get_protocol_members(protocol)) if not hasattr(context_type, member) and member not in annotated]
