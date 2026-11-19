"""``@node`` and ``subgraph()``, the two ways to make a node.

``@node`` turns a plain function into a node. Its typed signature is the node's interface::

    @node("upper", title="Uppercase", description="Makes every letter a capital letter.")
    def upper(text: str) -> str:
        return text.upper()

The result is a frozen ``FunctionDefinition``. It registers nothing, and it can still be called, so ``upper("a")``
is a unit test. A host offers the node by adding it to a ``NodeRegistry``.

The parameters are the node's inputs. The return type gives its outputs. ``-> str`` is one output, ``result``. A
frozen dataclass gives one output per field, unless it is declared with ``@host_type``: then it is a type like
``str``, and one output. ``Series[X]`` gives a list the graph can loop over, and ``Table[R]`` one whole table.

``@node`` checks the declaration when the module is imported. A parameter with no type, a default its type
refuses, or a ``raise Ask(...)`` in its body without ``asks=True`` raises ``NodeDeclarationError`` right there.

``subgraph()`` turns a graph record into a node instead, so a whole graph can be one step in another graph.

The rules for versions, hooks and asking are in the reference: ``python -m conductor.about the-node-contract``,
``versions-and-policies`` and ``field-hooks``.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing
from collections.abc import Callable, Mapping
from dataclasses import Field as DataclassField
from inspect import Parameter
from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol

from pydantic import JsonValue

from conductor._internal.base import REQUIRED, is_name
from conductor._internal.context import ContextParameter, is_context
from conductor._internal.field import DeclaredInputValidator, DeclaredOutputValidator, Field
from conductor._internal.problem import NodeDeclarationError, declaration_problems
from conductor._internal.record import GraphRecord, RecordValidator
from conductor._internal.types import Param, Type, fit, is_declared

from .policy import Deprecation, Policy
from .source import read_code
from .type import (
    FunctionDefinition,
    HookName,
    HookParameter,
    NodeDefinition,
    NodeFunction,
    SubgraphDefinition,
    Upgrade,
    gone_inputs,
    gone_outputs,
)


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
        NodeDeclarationError: When the decorator is applied, if the function or the arguments can't make a node.
    """
    number = _version_number("@node", id, title, description, version, replaces, upgrade)

    def make[**P, R](target: NodeFunction[P, R], /) -> FunctionDefinition[P, R]:
        try:
            if inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target):
                raise NodeDeclarationError(declaration_problems.async_run())
            inputs, context, outputs, outputs_dataclass = _read_signature(
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
            hooks: dict[HookName, Callable[..., Any] | None] = {
                "compute_added_inputs": compute_added_inputs,
                "compute_outputs": compute_outputs,
            }
            hook_parameters = {
                kind: _check_hook(hook, kind, inputs, adds_inputs=compute_added_inputs is not None)
                for kind, hook in hooks.items()
                if hook is not None
            }
            # An input a hook reads takes only a constant, since the hook reads it when the graph is compiled.
            for parameters in hook_parameters.values():
                for name, receives in parameters.items():
                    if receives == "value":
                        field = inputs[name]
                        inputs[name] = Field(
                            name,
                            Annotated[field.hint, Param(constant_only=True)],
                            field.default,
                            is_output=False,
                        )
            definition = FunctionDefinition(
                function=target,
                inputs=inputs,
                outputs=outputs,
                outputs_dataclass=outputs_dataclass,
                code_text=code_text,
                context=context,
                id=id,
                version=number,
                title=title,
                description=description,
                category=category,
                tags=tuple(tags),
                extra=MappingProxyType(dict(extra)),
                docs=docs,
                deprecation=deprecation,
                replaces=replaces,
                upgrade=upgrade or Upgrade(),
                policy=policy,
                compute_added_inputs=compute_added_inputs,
                compute_outputs=compute_outputs,
                hook_parameters=hook_parameters,
                asks=asks,
            )
            if replaces is not None:
                _raise_if_upgrade_does_not_fit(replaces, definition)
            return definition
        # Each fault is placed at the node's id, with its version.
        except NodeDeclarationError as error:
            raise error.within(id, number) from None

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
    number = _version_number("subgraph", id, title, description, version, replaces, upgrade)

    def make(record: GraphRecord | str | bytes, /) -> SubgraphDefinition:
        definition = SubgraphDefinition(
            graph=RecordValidator()(record),
            id=id,
            version=number,
            title=title,
            description=description,
            category=category,
            tags=tuple(tags),
            extra=MappingProxyType(dict(extra)),
            docs=docs,
            deprecation=deprecation,
            replaces=replaces,
            upgrade=upgrade or Upgrade(),
        )
        if replaces is not None:
            try:
                _raise_if_upgrade_does_not_fit(replaces, definition)
            except NodeDeclarationError as error:
                raise error.within(id, number) from None
        return definition

    return make


def _version_number(
    declared_with: str,
    id: str,
    title: str,
    description: str,
    version: int | None,
    replaces: NodeDefinition | None,
    upgrade: Upgrade | None,
) -> int:
    """Checks the arguments ``@node`` and ``subgraph`` share, and returns the node's version number.

    Args:
        declared_with: ``"@node"`` or ``"subgraph"``. Each message starts with it.
        id: The node's id.
        title: The node's title.
        description: The node's description.
        version: The version number given, or ``None``.
        replaces: The version this one replaces, or ``None``.
        upgrade: The upgrade given, or ``None``.

    Returns:
        ``version`` if it is given. Otherwise one more than the version replaced, or 1 for a first version.

    Raises:
        NodeDeclarationError: The id isn't a valid node id, the version isn't a whole number of 1 or more, the
            title or description is empty or isn't text, or ``replaces``,
            ``upgrade`` and ``version`` don't fit together. Its ``loc`` is the id, except for an id that isn't valid,
            which can't name a place.
    """
    # These are a node author's arguments, which a type checker may not have seen, so their types are checked here.
    if not isinstance(id, str) or not is_name(id):
        raise NodeDeclarationError(
            declaration_problems.invalid_node_id(declared_with=declared_with, id=id)
        )
    # True is an int to Python, and never a version
    if version is not None and (type(version) is not int or version < 1):
        raise NodeDeclarationError(
            declaration_problems.invalid_version(
                declared_with=declared_with, id=id, version=repr(version)
            ),
            id,
        )
    # the version comes first, so each fault below names it
    if version is not None:
        number = version
    elif replaces is not None:
        number = replaces.version + 1
    else:
        number = 1
    named: tuple[tuple[Literal["title", "description"], str], ...] = (
        ("title", title),
        ("description", description),
    )
    for name, text in named:
        if not isinstance(text, str) or not text.strip():
            raise NodeDeclarationError(
                declaration_problems.empty_title_or_description(
                    declared_with=declared_with, id=id, empty=name
                ),
                id,
                version=number,
            )
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
    return number


def _read_signature(
    function: Callable[..., Any],
    computed_outputs: bool,
) -> tuple[dict[str, Field], ContextParameter | None, dict[str, Field], type | None]:
    """Turns a function's signature into the node's inputs and outputs.

    A type variable, as in ``def keep[T](value: T) -> T``, stays open in the field's hint. A graph fills it in when
    it is compiled.

    Args:
        function: The function under ``@node``.
        computed_outputs: Whether a ``compute_outputs`` hook gives the outputs, so the return type gives none.

    Returns:
        Four things: the inputs by name; the ``Context`` parameter, or ``None`` if the first parameter isn't one;
        the outputs by name; and the frozen dataclass the function returns its outputs in, or ``None`` if it
        returns one value (a dataclass declared with ``@host_type`` among them), a mapping or nothing.

    Raises:
        NodeDeclarationError: A parameter or the return type is one a node can't have.
    """
    parameters = (
        inspect.signature(function).parameters
    )  # done first, so something that isn't a function gets Python's own "is not a callable object"
    hints = typing.get_type_hints(function, include_extras=True)
    inputs: dict[str, Field] = {}
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
    outputs: dict[str, Field]
    outputs_dataclass: type | None = None
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
        outputs = {}
    elif returns is None or returns is type(None):
        outputs = {}
    elif (
        isinstance(returned_class, type)
        and dataclasses.is_dataclass(returned_class)
        and not is_declared(returned_class)
    ):  # a dataclass declared with @host_type is a type, so it is one value
        if not returned_class.__dataclass_params__.frozen:  # type: ignore[attr-defined]
            raise NodeDeclarationError(declaration_problems.outputs_not_frozen())
        given = dict(
            zip(
                getattr(returned_class, "__parameters__", ()), typing.get_args(returns), strict=True
            )
        )  # what the signature fills in for each type variable of the class
        own = typing.get_type_hints(returned_class, include_extras=True)
        outputs = {
            f.name: DeclaredOutputValidator(
                f.name, Type(own[f.name]).bind(given).hint if given else own[f.name]
            )(_declared_default(f))
            for f in dataclasses.fields(returned_class)
        }
        outputs_dataclass = returned_class
    else:
        outputs = {"result": DeclaredOutputValidator("result", returns)(REQUIRED)}
    on_inputs = {variable for field in inputs.values() for variable in field.whole.variables}
    for field in (*inputs.values(), *outputs.values()):
        for variable in field.whole.variables:
            if variable.__constraints__:
                raise NodeDeclarationError(
                    declaration_problems.type_variable_constrained(variable=variable.__name__),
                    field.name,
                )
            if variable not in on_inputs:
                raise NodeDeclarationError(
                    declaration_problems.type_variable_only_in_return(variable=variable.__name__),
                    field.name,
                )
    return inputs, context, outputs, outputs_dataclass


def _declared_default(declared: DataclassField[Any]) -> Any:
    """Returns a dataclass field's default, or ``REQUIRED`` if it has none.

    When the node asks a person, this default is the answer proposed for that output.
    """
    if declared.default is not dataclasses.MISSING:
        return declared.default
    if declared.default_factory is not dataclasses.MISSING:
        return declared.default_factory()
    return REQUIRED


def _check_hook(
    hook: Callable[..., Any], kind: HookName, inputs: Mapping[str, Field], *, adds_inputs: bool
) -> dict[str, HookParameter]:
    """Checks a hook's parameters, and returns what each one gets.

    Each parameter is passed by name and names an input of the node. A parameter annotated ``type`` gets the type
    arriving on that input. The one exception is ``**added: type`` on ``compute_outputs``, which gets the type
    arriving on each added input.

    Args:
        hook: The hook.
        kind: Which hook it is.
        inputs: The node's declared inputs, by name.
        adds_inputs: Whether the node has a ``compute_added_inputs`` hook, so there are added inputs to read.

    Returns:
        What each parameter gets, by the parameter's name.

    Raises:
        NodeDeclarationError: A parameter can't be passed by name, asks for an arriving type in
            ``compute_added_inputs``, or asks for a value and names no declared input. Or it is ``**kwargs``
            anywhere except as ``**added: type`` on the ``compute_outputs`` of a node that adds inputs.
    """
    parameters: dict[str, HookParameter] = {}
    for name, parameter in inspect.signature(hook).parameters.items():
        arriving = parameter.annotation in (type, "type")
        if arriving and kind == "compute_added_inputs":
            raise NodeDeclarationError(
                declaration_problems.arriving_on_compute_added_inputs(parameter=name)
            )
        if parameter.kind is parameter.VAR_KEYWORD:
            if not (arriving and adds_inputs):
                raise NodeDeclarationError(
                    declaration_problems.hook_parameter_var_keyword(hook=kind, parameter=name)
                )
            parameters[name] = "added_arriving_types"
        elif (
            parameter.kind is not parameter.POSITIONAL_OR_KEYWORD
            and parameter.kind is not parameter.KEYWORD_ONLY
        ):
            raise NodeDeclarationError(
                declaration_problems.hook_parameter_not_by_name(hook=kind, parameter=name)
            )
        elif arriving:
            parameters[name] = "arriving_type"
        elif name not in inputs:
            raise NodeDeclarationError(
                declaration_problems.hook_parameter_unknown(
                    hook=kind, parameter=name, inputs=sorted(inputs)
                )
            )
        else:
            parameters[name] = "value"
    return parameters


def _raise_if_upgrade_does_not_fit(older: NodeDefinition, newer: NodeDefinition) -> None:
    """Checks the newer version's ``Upgrade`` against the fields of the two versions.

    Each rename has to start at a field the older version declares and end at one the newer version declares. Say
    version 1 takes ``name``, and version 2 takes ``first`` with ``Upgrade(renamed_inputs={"nmae": "first"})``. The
    typo renames nothing, so the rename is refused. A version with a ``compute_added_inputs`` hook may have inputs it
    doesn't declare, so a rename of its inputs isn't checked, and the same goes for ``compute_outputs`` and outputs.

    Then each input the older version had and the newer one lacks needs a rename or a ``rewrite``, which rewrites the
    constants and defaults. A dropped output needs a rename, since what has to move is the edges out of it.

    Raises:
        NodeDeclarationError: ``upgrade_renames_unknown``, naming each rename that starts or ends at no declared
            field. ``upgrade_missing``, naming each dropped field nothing explains.
    """
    upgrade = newer.upgrade
    adds_inputs, adds_outputs = _hooks_add_fields(older)
    unknown_old = [
        *(
            name
            for name in upgrade.renamed_inputs
            if not adds_inputs and name not in older.input_names
        ),
        *(
            name
            for name in upgrade.renamed_outputs
            if not adds_outputs and name not in older.output_names
        ),
    ]
    adds_inputs, adds_outputs = _hooks_add_fields(newer)
    unknown_new = [
        *(
            name
            for name in upgrade.renamed_inputs.values()
            if not adds_inputs and name not in newer.input_names
        ),
        *(
            name
            for name in upgrade.renamed_outputs.values()
            if not adds_outputs and name not in newer.output_names
        ),
    ]
    if unknown_old or unknown_new:
        raise NodeDeclarationError(
            declaration_problems.upgrade_renames_unknown(old=unknown_old, new=unknown_new)
        )
    unexplained = gone_outputs(older, newer)
    if newer.upgrade.rewrite is None:
        unexplained |= gone_inputs(older, newer)
    if unexplained:
        raise NodeDeclarationError(declaration_problems.upgrade_missing(gone=sorted(unexplained)))


def _hooks_add_fields(version: NodeDefinition) -> tuple[bool, bool]:
    """Returns whether a version's hooks may add inputs, and whether they may give outputs, beside those it declares.

    Only a function has hooks. A subgraph's inputs and outputs are all in its record.
    """
    match version:
        case FunctionDefinition():
            return version.compute_added_inputs is not None, version.compute_outputs is not None
        case SubgraphDefinition():
            return False, False
