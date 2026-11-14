"""``@node``: a plain function, or a graph record, becomes a frozen ``NodeDefinition``.

A node is a function. Its typed signature is its interface::

    @node("upper", title="Uppercase", description="Makes every letter a capital letter.")
    def upper(text: str) -> str:
        return text.upper()

``@node`` wraps the function into a value and registers nothing, so
``upper("a")`` is the unit test and a host decides what to offer by adding
definitions to a ``NodeRegistry`` (whose ``@registry.node`` takes the same
arguments and adds the node too). The ``title`` and ``description`` are what
a person reads for the node, and each node declares both; the docstring is
the developer's and never reaches a person. A field's label is its name made
readable, unless its ``Param(title=)`` says otherwise. ``extra`` is the host's own JSON about the node, which conductor keeps and
never reads.

One output is ``result``, of whatever type is returned, a ``dict`` included;
a frozen dataclass returned gives one output per field. ``Series[X]`` gives many values, each an item, and ``Table[R]`` one table; ``SKIPPED`` on an output
is a branch not taken. A node whose function may raise ``Ask`` to stop a call and ask a person says ``asks=True``.

A newer version of a node names the one it ``replaces``, itself a ``@node``
of the same id, and says with ``upgrade`` how a node saved at that one moves
here. A definition is one version, and each version names the one before it in
``replaces``::

    @node("split", title="Split", description="Splits the text into lines.")
    def split_v1(text: str, sep: str = "\\n") -> Series[str]: ...

    @node("split", title="Split", description="Splits the text into lines.", replaces=split_v1, upgrade=Upgrade(renamed_inputs={"sep": "separator"}))
    def split(text: str, separator: str = "\\n") -> Series[str]: ...

Two hooks shape a node from what it is given in one graph, each a plain
function called when a graph is compiled, never per call. A parameter of a hook
names an input of the node and receives the input's constant in the graph record (or its
default); a parameter annotated ``type`` receives the type arriving on the
input of that name by its edge (``Any`` until one does), and ``**added: type``
on ``compute_outputs`` receives the type arriving on each input
``compute_added_inputs`` added. A hook returns
``{name: type hint}``, read like a signature, or ``{name: (type hint, default)}``
for a field with a default, as pydantic's ``create_model`` takes one. A hook
that raises ``NodeError`` gives its own message, code and ``ctx`` to the
problem. ``compute_added_inputs`` adds
inputs, which the function takes as ``**kwargs``; ``compute_outputs`` gives
every output, and the function returns them as a mapping. A hook that raises
is a problem of the graph, not an exception.

A graph record can be a node too, a subgraph: ``subgraph("enrich", title="Enrich", description="...")(record)``. A run of a
graph that uses it calls the subgraph's nodes where the node is.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing
from collections.abc import Callable, Mapping
from dataclasses import Field as DataclassField
from inspect import Parameter
from types import MappingProxyType
from typing import Any, Protocol

from pydantic import JsonValue

from conductor._internal.base import REQUIRED, is_name
from conductor._internal.context import ContextParameter, is_context
from conductor._internal.field import Field
from conductor._internal.gate import declared_field, read_record
from conductor._internal.problem import NodeDeclarationError, declaration_problems
from conductor._internal.record import GraphRecord
from conductor._internal.types import Type

from .policy import Deprecation, Policy
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
    """What ``@node(...)`` returns: a decorator that makes a node of the function under it."""

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
    policy: Policy | None = None,
    deprecation: Deprecation | None = None,
    compute_added_inputs: Callable[..., Mapping[Any, Any]] | None = None,
    compute_outputs: Callable[..., Mapping[Any, Any]] | None = None,
    replaces: NodeDefinition | None = None,
    upgrade: Upgrade | None = None,
    asks: bool = False,
) -> NodeDecorator:
    """Make the function below it a node with the id ``id``::

        @node("upper", title="Uppercase", description="Makes every letter a capital letter.")
        def upper(text: str) -> str:
            return text.upper()

    It registers nothing: ``upper("a")`` still calls the function, and
    ``registry.add(upper)`` offers it. ``@registry.node`` takes the same
    arguments and adds the node too.

    - ``title`` and ``description`` are what a person reads for the node in
      a palette; the docstring is the developer's and never reaches them.
      ``docs`` is longer help, ``category`` and ``tags`` group it in a
      palette, and ``extra`` is the host's own JSON, kept and never read here.
    - ``policy`` says how the engine calls it (retries, a timeout, threads).
    - ``replaces`` is the version this one replaces, a ``@node`` of the same
      id, and ``upgrade`` says how a node saved at that version moves here.
      ``version`` is one past the version replaced, or 1, unless given: a host
      whose subgraphs have their own numbers names them. ``deprecation``
      marks this version going away.
    - ``compute_added_inputs`` and ``compute_outputs`` are hooks run when a
      graph is loaded, which shape the node from what is typed into it.
    - ``asks=True`` says the function may raise ``Ask`` to ask a person, who
      answers with its outputs. A raise from a node that doesn't say so fails
      the call (``ask_not_declared``).

    A graph record becomes a node, a subgraph, with ``subgraph``.
    """
    number = _version_number("@node", id, title, description, version, replaces, upgrade)

    def make[**P, R](target: NodeFunction[P, R], /) -> FunctionDefinition[P, R]:
        try:
            if inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target):
                raise NodeDeclarationError(declaration_problems.async_run())
            inputs, context, outputs, outputs_dataclass = _read_signature(target, compute_outputs is not None)
            takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in inspect.signature(target).parameters.values())
            if compute_added_inputs is not None and not takes_kwargs:
                raise NodeDeclarationError(declaration_problems.compute_added_inputs_without_var_keyword())
            if context is not None and policy is not None and policy.costs_nothing:
                raise NodeDeclarationError(declaration_problems.costs_nothing_takes_context(), context.name)
            if compute_added_inputs is None and takes_kwargs:
                raise NodeDeclarationError(declaration_problems.var_keyword_without_hook())
            hooks: dict[HookName, Callable[..., Any] | None] = {"compute_added_inputs": compute_added_inputs, "compute_outputs": compute_outputs}
            hook_parameters = {
                kind: _check_hook(hook, kind, inputs, adds_inputs=compute_added_inputs is not None)
                for kind, hook in hooks.items() if hook is not None
            }
            definition = FunctionDefinition(
                function=target, inputs=inputs, outputs=outputs, outputs_dataclass=outputs_dataclass, context=context,
                id=id, version=number, title=title, description=description,
                category=category, tags=tuple(tags), extra=MappingProxyType(dict(extra)), docs=docs, deprecation=deprecation,
                replaces=replaces, upgrade=upgrade or Upgrade(), policy=policy or Policy(),
                compute_added_inputs=compute_added_inputs, compute_outputs=compute_outputs, hook_parameters=hook_parameters,
                asks=asks,
            )
            if replaces is not None:
                _raise_if_upgrade_is_missing(replaces, definition)
            return definition
        except NodeDeclarationError as error:  # every fault of a declaration says which node and function it is on
            raise error.within(f"@node({id!r}) on {target.__name__}") from None

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
    """Make a graph record a node, a subgraph, with the id ``id``, so other graphs can use it as one node::

        enrich = subgraph("enrich", title="Enrich", description="Looks up and adds the company's details.")(record)

    The record is given as ``Graph.compile`` takes one: a ``GraphRecord`` or
    its JSON text. Text of the wrong shape raises ``InvalidCallError`` with
    ``invalid_record`` problems. The arguments mean what they mean on
    ``@node``, and a newer version that drops an input or output of the one it
    replaces needs an ``Upgrade`` saying where it went, as there (``upgrade_missing``).
    There is no ``policy`` and there are no hooks: the nodes inside
    the subgraph are called by their own. It registers nothing:
    ``registry.add(enrich)`` offers it, and ``registry.subgraph`` takes
    the same arguments and adds the node too.
    """
    number = _version_number("subgraph", id, title, description, version, replaces, upgrade)

    def make(record: GraphRecord | str | bytes, /) -> SubgraphDefinition:
        definition = SubgraphDefinition(
            graph=read_record(record), id=id, version=number, title=title, description=description, category=category,
            tags=tuple(tags), extra=MappingProxyType(dict(extra)), docs=docs, deprecation=deprecation, replaces=replaces,
            upgrade=upgrade or Upgrade(),
        )
        if replaces is not None:
            try:
                _raise_if_upgrade_is_missing(replaces, definition)
            except NodeDeclarationError as error:
                raise error.within(f"subgraph({id!r})") from None
        return definition

    return make


def _version_number(
    declared_with: str, id: str, title: str, description: str, version: int | None, replaces: NodeDefinition | None, upgrade: Upgrade | None,
) -> int:
    """Check the arguments ``@node`` and ``subgraph`` share, and return the node's version number.

    Args:
        declared_with: ``"@node"`` or ``"subgraph"``, to start each message with.
        id: The node's id.
        title: The node's title.
        description: The node's description.
        version: The version number given, or ``None``.
        replaces: The version this one replaces, or ``None``.
        upgrade: The upgrade given, or ``None``.

    Returns:
        ``version`` when given; else one more than the version replaced, or 1 for a first version.

    Raises:
        NodeDeclarationError: The id is not a node id, the title or description is empty, or ``replaces``,
            ``upgrade`` and ``version`` don't fit together.
    """
    if not is_name(id):
        raise NodeDeclarationError(declaration_problems.invalid_node_id(declared_with=declared_with, id=id))
    if not title.strip() or not description.strip():
        raise NodeDeclarationError(declaration_problems.empty_title_or_description(declared_with=declared_with, id=id, empty="title" if not title.strip() else "description"))
    if upgrade is not None and replaces is None:
        raise NodeDeclarationError(declaration_problems.upgrade_without_replaces(declared_with=declared_with, id=id))
    if replaces is not None and replaces.id != id:
        raise NodeDeclarationError(declaration_problems.replaces_another_node(declared_with=declared_with, id=id, replaces=repr(replaces)))
    number = version if version is not None else (replaces.version + 1 if replaces is not None else 1)
    if replaces is not None and number <= replaces.version:
        raise NodeDeclarationError(declaration_problems.version_not_after(declared_with=declared_with, id=id, version=number, replaced=replaces.version))
    return number


def _read_signature(
    function: Callable[..., Any], computed_outputs: bool,
) -> tuple[dict[str, Field], ContextParameter | None, dict[str, Field], type | None]:
    """Reads a function's signature.

    A type variable in it (``def keep[T](value: T) -> T``) stays in the field's hint, to be bound when a graph loads.

    Returns:
        The inputs by name, the ``Context`` parameter (``None`` when the first parameter is not one), the outputs
        by name, and the frozen dataclass the function returns its outputs as (``None`` when it returns one value, a
        mapping or nothing).

    Raises:
        NodeDeclarationError: A parameter or the return type is declared in a way a node can't have.
    """
    parameters = inspect.signature(function).parameters  # read first: for what is not a function it raises Python's "is not a callable object"
    hints = typing.get_type_hints(function, include_extras=True)
    inputs: dict[str, Field] = {}
    context: ContextParameter | None = None
    for position, (name, parameter) in enumerate(parameters.items()):
        # A Context anywhere else is read as an input below, and its type refuses to be read (context_misplaced).
        if position == 0 and name in hints and is_context(hints[name]) and parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
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
        inputs[name] = declared_field(name, hints[name], default)
        if inputs[name].param.exclusive_group is not None:
            raise NodeDeclarationError(declaration_problems.exclusive_group_on_input(), name)
    if "return" not in hints:
        raise NodeDeclarationError(declaration_problems.output_without_type())
    returns = hints["return"]
    outputs: dict[str, Field]
    outputs_dataclass: type | None = None
    returned_class = typing.get_origin(returns) or returns  # a generic dataclass of outputs is written with its arguments
    if computed_outputs or returns is None or returns is type(None):
        outputs = {}
    elif isinstance(returned_class, type) and dataclasses.is_dataclass(returned_class):
        if not returned_class.__dataclass_params__.frozen:  # type: ignore[attr-defined]
            raise NodeDeclarationError(declaration_problems.outputs_not_frozen())
        given = dict(zip(getattr(returned_class, "__parameters__", ()), typing.get_args(returns), strict=True))  # the class's variables as the signature names them
        own = typing.get_type_hints(returned_class, include_extras=True)
        outputs = {f.name: declared_field(f.name, Type(own[f.name]).bind(given).hint if given else own[f.name], _declared_default(f)) for f in dataclasses.fields(returned_class)}
        outputs_dataclass = returned_class
    else:
        outputs = {"result": Field("result", returns)}
    on_inputs = {variable for field in inputs.values() for variable in field.whole.variables}
    for field in (*inputs.values(), *outputs.values()):
        for variable in field.whole.variables:
            if variable.__constraints__:
                raise NodeDeclarationError(declaration_problems.type_variable_constrained(variable=variable.__name__), field.name)
            if variable not in on_inputs:
                raise NodeDeclarationError(declaration_problems.type_variable_only_in_return(variable=variable.__name__), field.name)
    return inputs, context, outputs, outputs_dataclass


def _declared_default(declared: DataclassField[Any]) -> Any:
    """A dataclass field's default, which a person asked for the output sees proposed; ``REQUIRED`` when it has none."""
    if declared.default is not dataclasses.MISSING:
        return declared.default
    if declared.default_factory is not dataclasses.MISSING:
        return declared.default_factory()
    return REQUIRED


def _check_hook(hook: Callable[..., Any], kind: HookName, inputs: Mapping[str, Field], *, adds_inputs: bool) -> dict[str, HookParameter]:
    """Reads a hook's parameters: each is passed by name and names an input of the node, but for ``**added: type``.

    Args:
        hook: The hook.
        kind: Which hook it is.
        inputs: The node's declared inputs, by name.
        adds_inputs: Whether the node has a ``compute_added_inputs`` hook, so there are added inputs to read.

    Returns:
        What each parameter receives, by the parameter's name.

    Raises:
        NodeDeclarationError: A parameter is positional; reads an arriving type in ``compute_added_inputs``; reads
            a value and names no declared input; or is ``**kwargs`` anywhere but as ``**added: type`` on the
            ``compute_outputs`` of a node that adds inputs.
    """
    parameters: dict[str, HookParameter] = {}
    for name, parameter in inspect.signature(hook).parameters.items():
        arriving = parameter.annotation in (type, "type")
        if arriving and kind == "compute_added_inputs":
            raise NodeDeclarationError(declaration_problems.arriving_on_compute_added_inputs(parameter=name))
        if parameter.kind is parameter.VAR_KEYWORD:
            if not (arriving and adds_inputs):
                raise NodeDeclarationError(declaration_problems.hook_parameter_var_keyword(hook=kind, parameter=name))
            parameters[name] = "added_arriving_types"
        elif parameter.kind is not parameter.POSITIONAL_OR_KEYWORD and parameter.kind is not parameter.KEYWORD_ONLY:
            raise NodeDeclarationError(declaration_problems.hook_parameter_not_by_name(hook=kind, parameter=name))
        elif arriving:
            parameters[name] = "arriving_type"
        elif name not in inputs:
            raise NodeDeclarationError(declaration_problems.hook_parameter_unknown(hook=kind, parameter=name, inputs=sorted(inputs)))
        else:
            parameters[name] = "value"
    return parameters


def _raise_if_upgrade_is_missing(older: NodeDefinition, newer: NodeDefinition) -> None:
    """An input or output of the replaced version that is gone from the newer one needs an upgrade to say where it
    went, for a function and a subgraph alike.

    A gone input is explained by a rename or a ``rewrite``, which rewrites what is typed in; a gone output only by a
    rename, since the edges out of it are what moves.
    """
    unexplained = gone_outputs(older, newer)
    if newer.upgrade.rewrite is None:
        unexplained |= gone_inputs(older, newer)
    if unexplained:
        raise NodeDeclarationError(declaration_problems.upgrade_missing(gone=sorted(unexplained)))
