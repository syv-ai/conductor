"""``@node`` and ``subgraph()``, the two ways to make a node.

``@node`` turns a plain function into a node. Its typed signature is the node's interface::

    @node("upper", title="Uppercase", description="Makes every letter a capital letter.")
    def upper(text: str) -> str:
        return text.upper()

The result is a frozen ``FunctionDefinition``. It registers nothing, and it can still be called, so ``upper("a")``
is a unit test. A host offers the node by adding it to a ``NodeRegistry``.

The parameters are the node's inputs. The return type gives its outputs. ``-> str`` is one output, ``result``. A
dataclass gives one output per field, unless it is declared with ``@host_type``: then it is a type like
``str``, and one output. ``Series[X]`` gives a list the graph can loop over, and ``Table[R]`` one whole table.

``@node`` checks the declaration when the module is imported. A parameter with no type, a default its type
refuses, or a ``raise Ask(...)`` in its body without ``asks=True`` raises ``NodeDeclarationError`` right there.

``subgraph()`` turns a graph record into a node instead, so a whole graph can be one step in another graph.

The rules for versions, hooks and asking are in the reference: ``python -m conductor.about the-node-contract``,
``versions-and-policies`` and ``hooks``.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing
from collections.abc import Callable, Mapping
from dataclasses import Field as DataclassField
from inspect import Parameter
from types import MappingProxyType
from typing import Annotated, Any, Protocol

from pydantic import JsonValue

from conductor._internal.base import REQUIRED
from conductor._internal.context import ContextParameter, is_context
from conductor._internal.port import (
    DeclaredInputValidator,
    DeclaredOutputValidator,
    Port,
    ReturnShape,
)
from conductor._internal.problem import NodeDeclarationError, declaration_problems
from conductor._internal.record import GraphRecord, RecordValidator
from conductor._internal.types import Param, Type, field_hints, fit, is_declared

from .hook import Hook
from .policy import Deprecation, Policy
from .source import read_code
from .type import (
    FunctionDefinition,
    NodeDefinition,
    NodeFunction,
    SubgraphDefinition,
    Upgrade,
    gone_inputs,
    gone_outputs,
    version_number,
)
from .validator import DeclarationValidator


class NodeDecorator(Protocol):
    """What ``@node(...)`` returns: a decorator that turns the function under it into a node."""

    def __call__[**P, R](self, function: NodeFunction[P, R], /) -> FunctionDefinition[P, R]: ...


def node(
    id: str,
    /,
    *,
    title: str,
    description: str,
    version: int | None = None,
    category: str = "",
    tags: list[str] | tuple[str, ...] = (),
    extra: Mapping[str, JsonValue] = MappingProxyType({}),
    docs: str | None = None,
    policy: Policy = Policy(),
    deprecation: Deprecation | None = None,
    compute_added_inputs: Callable[..., Mapping[Any, Any]] | None = None,
    compute_outputs: Callable[..., Mapping[Any, Any]] | None = None,
    replaces: NodeDefinition | None = None,
    upgrade: Upgrade | None = None,
    asks: bool = False,
) -> NodeDecorator:
    """Turns the function under it into a node with the id ``id``.

    ::

        @node("upper", title="Uppercase", description="Makes every letter a capital letter.")
        def upper(text: str) -> str:
            return text.upper()

    It registers nothing. ``upper("a")`` still calls the function, and ``registry.add(upper)`` offers the node.
    ``@registry.node`` takes the same arguments and also adds the node.

    Args:
        id: The node's id. A graph record names the node by it.
        title: The node's name, as a person reads it in a palette.
        description: What the node does, for a person. The function's docstring is for developers and is never
            shown.
        version: The version number. By default it is one more than the version ``replaces`` names, or 1.
        category: A group to show the node under in a palette.
        tags: Words to find the node by in a palette.
        extra: The host's own JSON about the node. Conductor keeps it and never looks at it.
        docs: Longer help text.
        policy: How the engine calls the node: how many calls at once, retries and a time limit.
        deprecation: A notice that this version is going away.
        compute_added_inputs: A hook that adds inputs, which the function takes as ``**kwargs``. It runs when a
            graph is compiled.
        compute_outputs: A hook that gives every output, when a graph is compiled. The function then returns its
            outputs as a mapping by name.
        replaces: The older version this one replaces, a ``@node`` with the same id.
        upgrade: How a graph that uses the older version moves to this one.
        asks: Whether the function may raise ``Ask`` to ask a person. Without it, a ``raise Ask(...)`` in the
            function's own body is refused here (``ask_not_declared_in_source``), and any other ``Ask`` fails its
            call (``ask_not_declared``).

    An input a hook reads takes only a constant, as if its ``Param`` said ``constant_only=True``. The hook reads it
    when the graph is compiled, before any run.

    Raises:
        NodeDeclarationError: Right away, ``invalid_declaration_argument`` if an argument isn't of the type the
            signature says, such as ``tags="text"``, or one of the codes for arguments that don't fit together. When
            the decorator is applied, if the function can't make a node.
    """
    shared = _shared_fields(
        "@node",
        {
            "id": id,
            "title": title,
            "description": description,
            "version": version,
            "category": category,
            "tags": tags,
            "extra": extra,
            "docs": docs,
            "policy": policy,
            "deprecation": deprecation,
            "compute_added_inputs": compute_added_inputs,
            "compute_outputs": compute_outputs,
            "replaces": replaces,
            "upgrade": upgrade,
            "asks": asks,
        },
    )

    def make[**P, R](target: NodeFunction[P, R], /) -> FunctionDefinition[P, R]:
        try:
            if inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target):
                raise NodeDeclarationError(declaration_problems.async_run())
            inputs, context, outputs, return_shape = _read_signature(
                target, compute_outputs is not None
            )
            code_text, raises_ask = read_code(target)
            if not asks and raises_ask:
                raise NodeDeclarationError(declaration_problems.ask_not_declared_in_source())
            takes_kwargs = any(
                p.kind is p.VAR_KEYWORD for p in inspect.signature(target).parameters.values()
            )
            if compute_added_inputs is not None and not takes_kwargs:
                raise NodeDeclarationError(
                    declaration_problems.compute_added_inputs_without_var_keyword()
                )
            if policy.costs_nothing and context is not None:
                raise NodeDeclarationError(
                    declaration_problems.costs_nothing_takes_context(), context.name
                )
            if compute_added_inputs is None and takes_kwargs:
                raise NodeDeclarationError(declaration_problems.var_keyword_without_hook())
            added_inputs_hook = None
            if compute_added_inputs is not None:
                added_inputs_hook = Hook.read(
                    compute_added_inputs, "compute_added_inputs", inputs, context, adds_inputs=True
                )
            outputs_hook = None
            if compute_outputs is not None:
                outputs_hook = Hook.read(
                    compute_outputs,
                    "compute_outputs",
                    inputs,
                    context,
                    adds_inputs=added_inputs_hook is not None,
                )
            # An input a hook reads takes only a constant, since the hook reads it when the graph is compiled.
            read_by_a_hook = {
                name
                for hook in (added_inputs_hook, outputs_hook)
                if hook is not None
                for name in hook.values_of
            }
            for name in read_by_a_hook:
                port = inputs[name]
                inputs[name] = Port(
                    name,
                    Annotated[port.hint, Param(constant_only=True)],
                    port.default,
                    is_output=False,
                )
            definition = FunctionDefinition(
                function=target,
                inputs=inputs,
                outputs=outputs,
                return_shape=return_shape,
                code_text=code_text,
                context=context,
                policy=policy,
                compute_added_inputs=added_inputs_hook,
                compute_outputs=outputs_hook,
                asks=asks,
                **shared,
            )
        # Each fault in the function is placed at the node's id, with its version.
        except NodeDeclarationError as error:
            raise error.within(id, shared["version"]) from None
        _raise_if_upgrade_does_not_fit(definition)
        return definition

    return make


def subgraph(
    id: str,
    /,
    *,
    title: str,
    description: str,
    version: int | None = None,
    category: str = "",
    tags: list[str] | tuple[str, ...] = (),
    extra: Mapping[str, JsonValue] = MappingProxyType({}),
    docs: str | None = None,
    deprecation: Deprecation | None = None,
    replaces: NodeDefinition | None = None,
    upgrade: Upgrade | None = None,
) -> Callable[[GraphRecord | str | bytes], SubgraphDefinition]:
    """Turns a graph record into a node with the id ``id``, so other graphs can use the whole graph as one step.

    ::

        enrich = subgraph("enrich", title="Enrich", description="Looks up and adds the company's details.")(record)

    The record can be a ``GraphRecord`` or its JSON text, as for ``Graph.compile``. Text of the wrong shape raises
    ``InvalidCallError`` with ``invalid_record`` problems.

    The arguments mean the same as on ``@node``. There is no ``policy`` and there are no hooks, since the nodes
    inside the subgraph are called by their own policies. Like ``@node``, it registers nothing:
    ``registry.add(enrich)`` offers it, and ``registry.subgraph`` takes the same arguments and also adds it.

    Raises:
        NodeDeclarationError: If the arguments can't make a node, for example a newer version that drops an input
            with no ``Upgrade`` saying where it went (``upgrade_missing``), or an ``Upgrade`` that renames a graph
            input or output one of the two records doesn't have (``upgrade_renames_unknown``).
    """
    shared = _shared_fields(
        "subgraph",
        {
            "id": id,
            "title": title,
            "description": description,
            "version": version,
            "category": category,
            "tags": tags,
            "extra": extra,
            "docs": docs,
            "deprecation": deprecation,
            "replaces": replaces,
            "upgrade": upgrade,
        },
    )

    def make(record: GraphRecord | str | bytes, /) -> SubgraphDefinition:
        definition = SubgraphDefinition(graph=RecordValidator()(record), **shared)
        _raise_if_upgrade_does_not_fit(definition)
        return definition

    return make


def _shared_fields(declared_with: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Checks the arguments of ``@node`` or ``subgraph``, and returns the fields every version of a node has.

    Each argument is checked on its own first, by ``DeclarationValidator``. Then the version number is worked out,
    and ``version``, ``replaces`` and ``upgrade`` are checked together. A kind's own arguments, such as ``@node``'s
    ``policy``, are checked here too, but only the shared fields are returned.

    Args:
        declared_with: ``"@node"`` or ``"subgraph"``. Each message starts with it.
        arguments: Every argument the caller was given, by name.

    Returns:
        The fields of ``_Version`` by name, ready to pass to ``FunctionDefinition`` or ``SubgraphDefinition``. The
        version is ``version`` if it is given, else one more than the version replaced, or 1 for a first version.

    Raises:
        NodeDeclarationError: An argument is refused, or ``replaces``, ``upgrade`` and ``version`` don't fit
            together. Its ``loc`` is the id, unless the id itself is refused.
    """
    read = DeclarationValidator(declared_with)(arguments)
    id, replaces, upgrade = read["id"], read["replaces"], read["upgrade"]
    # the version comes first, so each fault below names it
    number = version_number(read["version"], replaces)
    if upgrade is not None and replaces is None:
        raise NodeDeclarationError(
            declaration_problems.upgrade_without_replaces(declared_with=declared_with, id=id),
            id,
            version=number,
        )
    if replaces is not None and replaces.id != id:
        raise NodeDeclarationError(
            declaration_problems.replaces_another_node(
                declared_with=declared_with, id=id, replaces=repr(replaces)
            ),
            id,
            version=number,
        )
    if replaces is not None and number <= replaces.version:
        raise NodeDeclarationError(
            declaration_problems.version_not_after(
                declared_with=declared_with, id=id, version=number, replaced=replaces.version
            ),
            id,
            version=number,
        )
    return {
        "id": id,
        "version": number,
        "title": read["title"],
        "description": read["description"],
        "category": read["category"],
        "tags": tuple(read["tags"]),
        "extra": MappingProxyType(dict(read["extra"])),
        "docs": read["docs"],
        "deprecation": read["deprecation"],
        "replaces": replaces,
        "upgrade": upgrade or Upgrade(),
    }


def _read_signature(
    function: Callable[..., Any],
    computed_outputs: bool,
) -> tuple[dict[str, Port], ContextParameter | None, dict[str, Port], ReturnShape]:
    """Turns a function's signature into the node's inputs and outputs.

    A type variable, as in ``def keep[T](value: T) -> T``, stays open in the port's hint. A graph fills it in when
    it is compiled.

    Args:
        function: The function under ``@node``.
        computed_outputs: Whether a ``compute_outputs`` hook gives the outputs, so the return type gives none.

    Returns:
        Four things: the inputs by name; the ``Context`` parameter, or ``None`` if the first parameter isn't one;
        the outputs by name; and how the function returns them (``ReturnShape``).

    Raises:
        NodeDeclarationError: A parameter or the return type is one a node can't have.
    """
    parameters = (
        inspect.signature(function).parameters
    )  # done first, so something that isn't a function gets Python's own "is not a callable object"
    hints = typing.get_type_hints(function, include_extras=True)
    inputs: dict[str, Port] = {}
    context: ContextParameter | None = None
    for position, (name, parameter) in enumerate(parameters.items()):
        # A Context anywhere else is taken as an input below, and its type refuses that (context_misplaced).
        if (
            position == 0
            and name in hints
            and is_context(hints[name])
            and parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ):
            needs = typing.get_args(hints[name])[0]
            if not isinstance(needs, type):
                raise NodeDeclarationError(declaration_problems.context_not_a_class(), name)
            context = ContextParameter(name, needs)
            continue
        if parameter.kind is parameter.VAR_KEYWORD:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.POSITIONAL_ONLY):
            raise NodeDeclarationError(declaration_problems.parameter_not_by_name(), name)
        if name not in hints:
            raise NodeDeclarationError(declaration_problems.input_without_type(), name)
        default = REQUIRED if parameter.default is Parameter.empty else parameter.default
        inputs[name] = DeclaredInputValidator(name, hints[name])(default)
    if "return" not in hints:
        raise NodeDeclarationError(declaration_problems.output_without_type())
    returns = hints["return"]
    outputs: dict[str, Port]
    returned_as: ReturnShape
    returned_class = (
        typing.get_origin(returns) or returns
    )  # a generic dataclass of outputs is written with its type arguments, like Pair[str]
    if computed_outputs:
        # Asked by the rule an edge is checked by: may a mapping of outputs pass where the return type is declared?
        # It calls fit itself, since Type(Mapping) is refused: Mapping has no @host_type.
        if not fit(returns, dict[str, Any], {}, False):
            raise NodeDeclarationError(
                declaration_problems.outputs_not_a_mapping(returns=Type(returns).text)
            )
        outputs, returned_as = {}, "mapping"
    elif returns is None or returns is type(None):
        outputs, returned_as = {}, "nothing"
    elif (
        isinstance(returned_class, type)
        and dataclasses.is_dataclass(returned_class)
        and not is_declared(returned_class)
    ):  # a dataclass declared with @host_type is a type, so it is one value
        # each field with what the signature fills in for the class's type variables, as in Pair[int]
        own = field_hints(returns)
        outputs = {
            f.name: DeclaredOutputValidator(f.name, own[f.name])(_declared_default(f))
            for f in dataclasses.fields(returned_class)
        }
        returned_as = returned_class
    else:
        outputs, returned_as = (
            {"result": DeclaredOutputValidator("result", returns)(REQUIRED)},
            "value",
        )
    on_inputs = {variable for port in inputs.values() for variable in port.whole.variables}
    for port in (*inputs.values(), *outputs.values()):
        for variable in port.whole.variables:
            if variable.__constraints__:
                raise NodeDeclarationError(
                    declaration_problems.type_variable_constrained(variable=variable.__name__),
                    port.name,
                )
            if variable not in on_inputs:
                raise NodeDeclarationError(
                    declaration_problems.type_variable_only_in_return(variable=variable.__name__),
                    port.name,
                )
    return inputs, context, outputs, returned_as


def _declared_default(declared: DataclassField[Any]) -> Any:
    """Returns a dataclass field's default, or ``REQUIRED`` if it has none.

    When the node asks a person, this default is the answer proposed for that output.
    """
    if declared.default is not dataclasses.MISSING:
        return declared.default
    if declared.default_factory is not dataclasses.MISSING:
        return declared.default_factory()
    return REQUIRED


def _raise_if_upgrade_does_not_fit(newer: NodeDefinition) -> None:
    """Checks a version's ``Upgrade`` against its ports and those of the version it replaces.

    A first version, which replaces nothing, has nothing to check.

    Each rename has to start at a port the older version declares and end at one the newer version declares. Say
    version 1 takes ``name``, and version 2 takes ``first`` with ``Upgrade(renamed_inputs={"nmae": "first"})``. The
    typo renames nothing, so the rename is refused. A version with a ``compute_added_inputs`` hook may have inputs it
    doesn't declare, so a rename of its inputs isn't checked, and the same goes for ``compute_outputs`` and outputs.

    Then each input the older version had and the newer one lacks needs a rename or a ``rewrite``, which rewrites the
    constants and defaults. A dropped output needs a rename, since what has to move is the edges out of it.

    Raises:
        NodeDeclarationError: ``upgrade_renames_unknown``, naming each rename that starts or ends at no declared
            port. ``upgrade_missing``, naming each dropped port nothing explains. Its ``loc`` is the id.
    """
    older = newer.replaces
    if older is None:
        return
    upgrade = newer.upgrade
    unknown_old = [
        *(
            name
            for name in upgrade.renamed_inputs
            if older.compute_added_inputs is None and name not in older.input_names
        ),
        *(
            name
            for name in upgrade.renamed_outputs
            if older.compute_outputs is None and name not in older.output_names
        ),
    ]
    unknown_new = [
        *(
            name
            for name in upgrade.renamed_inputs.values()
            if newer.compute_added_inputs is None and name not in newer.input_names
        ),
        *(
            name
            for name in upgrade.renamed_outputs.values()
            if newer.compute_outputs is None and name not in newer.output_names
        ),
    ]
    if unknown_old or unknown_new:
        raise NodeDeclarationError(
            declaration_problems.upgrade_renames_unknown(old=unknown_old, new=unknown_new),
            newer.id,
            version=newer.version,
        )
    unexplained = gone_outputs(older, newer)
    if newer.upgrade.rewrite is None:
        unexplained |= gone_inputs(older, newer)
    if unexplained:
        raise NodeDeclarationError(
            declaration_problems.upgrade_missing(gone=sorted(unexplained)),
            newer.id,
            version=newer.version,
        )
