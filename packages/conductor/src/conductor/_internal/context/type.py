"""``Context[X]``, the type of a node's first parameter when the node takes the run's context.

Here a host passes the person running the graph and an HTTP client, and a node signs a text with the person's
name::

    @dataclass(frozen=True)
    class HostContext:
        caller: Caller
        http_client: HttpClient

    registry = NodeRegistry(context_type=HostContext)

    @registry.node("sign", title="Sign", description="Signs a text with the name of the person running it.")
    def sign(ctx: Context[HostContext], text: str) -> str:
        return f"{text}, {ctx.caller.name}"

    graph.run({"text": "Done"}, context=HostContext(caller=Caller("Ida"), http_client=client))

``ctx`` is not an input of the node. No edge reaches it, the graph record never mentions it, and a run never saves
it. It plays no part in whether a call's earlier outcome is reused, either. A unit test just calls
``sign(host_context, "Done")``.

A node may name the registry's class, a base of it, or a ``Protocol`` the class satisfies. A node that doesn't
know the host's class, such as a pre-built node, names a ``Protocol`` with the members it uses::

    class HasHttpClient(Protocol):
        @property
        def http_client(self) -> HttpClient: ...

Write each member as a read-only property. A frozen dataclass's fields can't be set, so a type checker says it
doesn't satisfy a Protocol whose member is a plain attribute.

When a graph is compiled, each node's ``Context[X]`` is checked against the registry's class
(``ContextParameter.fit``). For a class, the registry's class must be it or a subclass of it. For a Protocol, the
registry's class must have every member the Protocol declares, by name. The members' types aren't checked: an
``http_client: str`` still has the name ``HasHttpClient`` asks for. Whether the types fit is a job for the host's
type checker.
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
    """The metadata that ``Context[X]`` adds to ``X``: ``Context[X]`` is ``Annotated[X, _Context()]``.

    ``@node`` takes it off a node's first parameter. Anywhere else in a type hint, pydantic meets it while building
    a schema, and it refuses.
    """

    def __get_pydantic_core_schema__(
        self, source: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        """Refuses a ``Context`` used as an input's or output's type.

        That is a ``Context`` on a later or keyword-only parameter, inside ``| None`` or a list, or as the return
        type.

        Raises:
            NodeDeclarationError: Always, with ``context_misplaced``.
        """
        raise NodeDeclarationError(declaration_problems.context_misplaced())


#: ``Context[X]``: the type of a node's first parameter when it takes the run's context, the object the host passes
#: as ``context=``. To a type checker, the parameter is just an ``X``.
Context = Annotated[_T, _Context()]


def is_context(hint: Any) -> bool:
    """Whether ``hint`` is ``Context[X]`` for some ``X``."""
    return typing.get_origin(hint) is Annotated and any(
        isinstance(metadata, _Context) for metadata in typing.get_args(hint)[1:]
    )


@dataclass(frozen=True, slots=True)
class ContextParameter:
    """A node's first parameter, when the node takes the run's context.

    For ``sign(ctx: Context[HostContext], text: str)``, it is ``ContextParameter("ctx", HostContext)``.

    Attributes:
        name: The parameter's name. No input of the node may have the same name.
        cls: The class inside ``Context[...]``: the registry's class, a base of it, or a ``Protocol``.
    """

    name: str
    cls: type

    def fit(self, node_id: str, context_type: type | None) -> Problem | None:
        """Checks whether the registry's context class, ``context_type``, is what this parameter asks for.

        With ``HostContext`` from above, ``sign(ctx: Context[HostContext], text: str)`` fits ``HostContext`` and any
        subclass of it. ``fetch(ctx: Context[HasHttpClient], url: str)`` fits any class with an ``http_client``,
        whatever its type. On a registry that names no class, neither fits.

        Args:
            node_id: The node this parameter belongs to, for the problem's ``loc``.
            context_type: The registry's ``context_type``, or ``None`` when it names none.

        Returns:
            ``None`` when it fits. Otherwise a ``context_does_not_fit`` problem at ``(node_id, name)``, which says
            why: the class isn't ``cls`` or a subclass of it, it lacks members of the Protocol ``cls``
            (``context_does_not_fit_missing_member``), or the registry names no class (``context_does_not_fit_no_context_type``).
        """
        needs = self.cls.__name__
        if context_type is None:
            return compile_problems.context_does_not_fit_no_context_type(
                node_id, self.name, needs=needs
            )
        if not is_protocol(self.cls):
            return (
                None
                if issubclass(context_type, self.cls)
                else compile_problems.context_does_not_fit(
                    node_id, self.name, needs=needs, context_type=context_type.__name__
                )
            )
        lacks = _lacks(context_type, self.cls)
        return (
            compile_problems.context_does_not_fit_missing_member(
                node_id, self.name, needs=needs, context_type=context_type.__name__, lacks=lacks
            )
            if lacks
            else None
        )


def _lacks(context_type: type, protocol: type) -> list[str]:
    """Returns the members of ``protocol`` that the class ``context_type`` doesn't have, or ``[]`` if it has them all.

    With ``HostContext`` from above, ``HasHttpClient`` gives ``[]``, and a Protocol with a member ``database``
    gives ``["database"]``. The class has a member if it has an attribute, method or property of that name, or
    annotates a field of that name, such as a dataclass field with no default. The members' types aren't compared,
    as the module docstring says.
    """
    annotated = typing.get_type_hints(context_type)
    return [
        member
        for member in sorted(get_protocol_members(protocol))
        if not hasattr(context_type, member) and member not in annotated
    ]
