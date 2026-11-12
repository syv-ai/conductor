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
is a branch not taken. A node that may stop a call to ask a person says so: ``asks=True`` when its own function raises
``Ask``, or a parameter ``Annotated[bool, Asked(approval)]`` filled by a function that may raise it.

A newer version of a node names the one it ``replaces``, itself a ``@node``
of the same id, and says with ``upgrade`` how a node saved at that one moves
here. A definition is one version, and each version names the one before it in
``replaces``::

    @node("split", title="Split", description="Splits the text into lines.")
    def split_v1(text: str, sep: str = "\\n") -> Series[str]: ...

    @node("split", title="Split", description="Splits the text into lines.", replaces=split_v1, upgrade=Upgrade(renamed_inputs={"sep": "separator"}))
    def split(text: str, separator: str = "\\n") -> Series[str]: ...

Two hooks shape a node from what it is given in one graph, each a plain
function run when a graph is loaded, never per call. A parameter of a hook
names an input of the node and receives the value typed into it (or its
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

A graph record can be a node too, a subgraph: ``subgraph("enrich", title="Enrich", description="...")(record)``. Loading a
graph that uses it puts the subgraph's nodes in its place.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import typing
from collections.abc import Callable, Mapping
from dataclasses import Field as DataclassField
from dataclasses import dataclass
from inspect import Parameter
from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol

from pydantic import ConfigDict, JsonValue
from pydantic import Field as PydanticField

from conductor._internal.address import is_name
from conductor._internal.ask import Asked
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.context import ContextParameter, is_context
from conductor._internal.field import Field, FieldDescription
from conductor._internal.gate import AskedValueDoor, RefusedError, declared_field, read_record
from conductor._internal.gate.returned import (
    HookOutputs,
    NoOutputs,
    OneResult,
    OutputsClass,
    ReturnShape,
)
from conductor._internal.graph_record import GraphRecord
from conductor._internal.markers import REQUIRED
from conductor._internal.problem import InvalidCallError, NodeDeclarationError
from conductor._internal.problem import declarations as declaration
from conductor._internal.text import qualified_name
from conductor._internal.value_type import Type, split_hint


class NodeFunction[**P, R](Protocol):
    """A function a node is made of: called with its own signature, and named where it was written.

    Every function, method and static method has these names, so a node says where it was declared from them.
    """

    __name__: str
    __module__: str
    __qualname__: str

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

#: The hooks a node may have, by the keyword ``@node`` takes each as.
HookName = Literal["compute_added_inputs", "compute_outputs"]

#: What a hook's parameter receives: the ``value`` typed into the input it names; the ``arriving_type`` on that input;
#: or, as ``**name: type``, the ``added_arriving_types``, the type arriving on each input ``compute_added_inputs`` added.
HookParameter = Literal["value", "arriving_type", "added_arriving_types"]


class Policy(ConductorModel):
    """How the engine calls a node, set by its author: ``@node("fetch", policy=Policy(retries=3))``.

    ``concurrency`` calls are made at once, on threads; with 1 and no
    ``attempt_timeout`` the node is called in order, on the leg's own thread.
    A call that raises ``ExternalError``, or one of the ``retry_on`` classes
    of the node's own client, is tried ``retries`` more times, waiting from
    ``retry_delay`` seconds (the reference has the schedule); anything else
    fails at once, since repeating a computation repeats its failure. A
    ``NodeError`` other than ``ExternalError``, an output its type refuses
    and an ``Ask`` are never tried again, whatever ``retry_on`` names. An
    attempt that runs past ``attempt_timeout`` fails the unit, finally: its
    thread can't be stopped, and a second call beside it would repeat its
    effect. ``reuse_when_inputs_match=False`` is for a node that reads the
    world, not only its inputs: ``run.again`` calls it for every item.
    ``costs_nothing=True`` is for a node whose call costs nothing and touches
    nothing outside the run (one that hands on an upload's files, splits a
    table, or returns a list given to it): ``run.resume(only_costs_nothing=True)``
    calls only such nodes, so a host learns how many calls the costly nodes
    will have (``run.progress``) before any of them is made.

    It is also what an editor reads of a node's policy: as JSON it leaves ``retry_on`` out, since a class has no
    JSON form. Its values are read strictly: a whole number is no yes or no and no text that spells one.

    Attributes:
        concurrency: How many calls are made at once; at least 1.
        retries: How many more times a call that failed outside the node's control is tried; at least 0.
        retry_delay: The seconds the first retry waits; finite and at least 0.
        attempt_timeout: The seconds one attempt may take; finite and above 0, or ``None`` for no clock.
        retry_on: The exception classes of the node's own client that are tried again like an ``ExternalError``.
        reuse_when_inputs_match: Whether ``run.again`` reuses a result when a call receives what it received before.
        costs_nothing: Whether a call costs nothing, so ``run.resume(only_costs_nothing=True)`` makes it. A node
            that takes the run's context is refused it.
    """

    model_config = ConfigDict(strict=True)

    concurrency: int = PydanticField(default=1, ge=1)
    retries: int = PydanticField(default=0, ge=0)
    retry_delay: float = PydanticField(default=1.0, ge=0, allow_inf_nan=False)
    attempt_timeout: float | None = PydanticField(default=None, gt=0, allow_inf_nan=False)
    retry_on: tuple[type[BaseException], ...] = PydanticField(default=(), exclude=True)
    reuse_when_inputs_match: bool = True
    costs_nothing: bool = False

    @functools.cached_property
    def threaded(self) -> bool:
        """Whether a call goes to a thread: several at once, or a clock on each attempt."""
        return self.concurrency > 1 or self.attempt_timeout is not None


class Deprecation(ConductorModel):
    """A notice that a version of a node is going away, which an editor shows beside the node.

    A graph that uses a deprecated node is not wrong, so this is never a ``Problem``. Every part is optional prose.

    Attributes:
        header: The notice in one line, as a badge or a title shows it: ``"Use version 2"``.
        description: Why the version is going away, in a sentence or two.
        alternative: The id of the node to use instead.
        migration: How to move a graph off this version.
    """

    header: str | None = None
    description: str | None = None
    alternative: str | None = None
    migration: str | None = None


@dataclass(frozen=True)
class Upgrade:
    """How a node saved at the version a newer one replaces becomes the newer one.

    Given on the newer version's ``@node``, beside ``replaces``; required only
    when a field name of the replaced version is gone from the newer one. A
    gone input is explained by a rename or ``rewrite``, a gone output by a rename.
    Read only by ``registry.upgrade``: loading never upgrades, so a graph
    saved at an old version runs that version.

    ``renamed_inputs`` moves each input's value, default and edges to the new
    name before ``rewrite`` runs. ``rewrite`` takes what is typed into the
    node, its values and defaults as one dict, and returns it for the newer
    version: a key that was a default stays one, any other key is a value.
    ``renamed_outputs`` moves every edge that read the old name, or a column
    of it.
    """

    renamed_inputs: Mapping[str, str] = dataclasses.field(default_factory=dict[str, str])
    renamed_outputs: Mapping[str, str] = dataclasses.field(default_factory=dict[str, str])
    rewrite: Callable[[dict[str, JsonValue]], Mapping[str, Any]] | None = None


@dataclass(frozen=True, slots=True)
class AskedParameter:
    """A node's ``Asked`` parameter, as ``@node`` reads it from the signature.

    For ``approve(invoice: Invoice, ok: Annotated[bool, Asked(approval)])``, with ``approval(invoice: Invoice) -> bool``,
    it is the parameter ``ok``, the function ``approval``, the inputs ``("invoice",)`` and no context.

    Attributes:
        field: The parameter as a field: its name, its type and its ``Param``, the type an answer is read by.
        function: The function that gives it its value.
        inputs: The node's inputs the function takes, by name.
        context: The name the function takes the node's context by, or ``None`` when it takes none.
        door: Reads what the function returns by the parameter's type (``invalid_asked_value``).
    """

    field: Field
    function: Callable[..., Any]
    inputs: tuple[str, ...]
    context: str | None
    door: AskedValueDoor


@dataclass(frozen=True, eq=False, kw_only=True)
class _Version:
    """What every version of a node has, whether it is a function or a subgraph.

    Never made on its own: a version is a ``FunctionDefinition`` or a ``SubgraphDefinition``, the two kinds of
    ``NodeDefinition``.

    Attributes:
        id: The id a graph names it by. Stored in every graph that uses it, so changing it is a data migration, not
            a rename.
        version: Which version of the node this is, counted from 1; a graph pins it.
        title: What a person reads for the node, as its author declared it.
        description: What it does, in a sentence for a person, as its author declared it; never the docstring.
        category: Where an editor lists it; a plain string whose titles the host keeps.
        tags: The host's labels for finding it, as given.
        extra: The host's own JSON about the node: kept and described, never read here.
        docs: Long-form markdown for the node's help, when the description isn't all there is to say.
        deprecation: Set when the node is on its way out: what to tell a person, and what to use instead.
        replaces: The version before this one, under the same id; ``None`` for a first version. It may be of the
            other kind.
        upgrade: How a node saved at ``replaces`` moves to this version; read only when ``replaces`` names one.
    """

    id: str
    version: int
    title: str
    description: str
    category: str = ""
    tags: tuple[str, ...] = ()
    extra: Mapping[str, JsonValue] = dataclasses.field(default_factory=dict[str, JsonValue])
    docs: str | None = None
    deprecation: Deprecation | None = None
    replaces: NodeDefinition | None = None
    upgrade: Upgrade = Upgrade()

    def __repr__(self) -> str:
        parts = [f"id={self.id!r}", f"version={self.version}", f"title={self.title!r}"]
        if self.category:
            parts.append(f"category={self.category!r}")
        if self.tags:
            parts.append(f"tags={self.tags!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    @functools.cached_property
    def _replaced(self) -> tuple[NodeDefinition, ...]:
        """Each version this one replaces, newest first."""
        return () if self.replaces is None else self.replaces.version_chain

    @functools.cached_property
    def versions(self) -> tuple[int, ...]:
        """Every version number, newest first."""
        return (self.version, *(step.version for step in self._replaced))

    def described_as[D: FunctionNodeDescription | SubgraphDescription](self, record: type[D], **more: Any) -> D:
        """Describes this version as an editor's palette shows it.

        Args:
            record: The kind of description to make.
            **more: What that kind holds beside what every version has: its fields, and a function's policy.
        """
        return record(
            id=self.id, version=self.version, versions=list(self.versions), title=self.title, description=self.description,
            category=self.category, tags=list(self.tags), extra=dict(self.extra), docs=self.docs, deprecation=self.deprecation, **more,
        )


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)  # the version's own __repr__
class FunctionDefinition[**P, R](_Version):
    """A version of a node that is a function: what it takes, what it gives, and how to call it.

    Frozen, and callable with the function's own signature, so a node is tested by calling it. ``@node`` on a
    ``def`` makes one; a host reads it from ``registry[id]`` or by looping over a registry.

    Attributes:
        function: What each call runs.
        inputs: Each input by name, as its signature declares it. The ``Context`` parameter is not an input: it is
            ``context``. An input a hook adds is not here either; a compiled graph's node has it.
        outputs: Each output by name: ``result`` for one value, a field each for a frozen dataclass returned.
        returns: What the function returns: one ``result``, a frozen dataclass of outputs, the mapping of outputs a
            ``compute_outputs`` hook gave, or nothing.
        policy: How it runs: how many calls at once, its retries and time limit, whether it reuses earlier work.
        compute_added_inputs: A hook run at load that adds inputs from the values typed into it; the function takes
            them as ``**kwargs``.
        compute_outputs: A hook run at load that gives every output, from the values typed in and the types
            arriving; the function returns them as a mapping.
        context: The first parameter when it is ``Context[X]``: the run's context, passed first by position on each
            call. ``None`` for a node that takes no context.
        hook_parameters: Each hook's parameters, by the hook's name (``compute_added_inputs``, ``compute_outputs``)
            and then the parameter's: what the parameter receives. Read from the hooks' signatures once, at
            ``@node``.
        generic: Whether an input's type holds a type variable, which a graph binds from its edges as it loads.
        asks: Whether the function itself may raise ``Ask``, as ``@node(asks=True)`` declares; the answer is then
            its outputs.
        asked: The parameter filled by a function that may raise ``Ask`` (``Annotated[bool, Asked(approval)]``), or
            ``None``. It is not in ``inputs``: no edge or constant reaches it.
    """

    function: NodeFunction[P, R]
    inputs: Mapping[str, Field]
    outputs: Mapping[str, Field]
    returns: ReturnShape
    policy: Policy = Policy()
    compute_added_inputs: Callable[..., Mapping[Any, Any]] | None = None
    compute_outputs: Callable[..., Mapping[Any, Any]] | None = None
    context: ContextParameter | None = None
    hook_parameters: Mapping[HookName, Mapping[str, HookParameter]] = dataclasses.field(default_factory=dict[HookName, Mapping[str, HookParameter]])
    generic: bool = False
    asks: bool = False
    asked: AskedParameter | None = None

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the function, as a test or a host would."""
        return self.function(*args, **kwargs)

    @property
    def version_chain(self) -> tuple[NodeDefinition, ...]:
        """This version, then each version it replaces, newest first."""
        return (self, *self._replaced)

    def version_of(self, version: int) -> NodeDefinition:
        """This node at ``version``, walking back through what each version replaces; a ``KeyError`` naming the versions
        there are when it has none. ``version in definition.versions`` asks first."""
        return _version_of(self.version_chain, version)

    def _where_declared(self) -> str:
        """Returns where this node was written, for a message that names it: its function's module and name.

        For ``def shout(...)`` in the module ``readme``, it is ``readme.shout``.
        """
        return qualified_name(self.function)

    def describe(self) -> FunctionNodeDescription:
        """The node as an editor's palette shows it."""
        return self.described_as(
            FunctionNodeDescription, policy=self.policy,
            inputs=[field.description for field in self.inputs.values()], outputs=[field.description for field in self.outputs.values()],
            takes_added_inputs=self.compute_added_inputs is not None, computes_outputs=self.compute_outputs is not None,
            asks=self.asks or self.asked is not None, asked=None if self.asked is None else self.asked.field.description,
        )


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)  # the version's own __repr__
class SubgraphDefinition(_Version):
    """A graph record registered as a node, one version of it: ``subgraph(...)(record)`` makes one. Loading a graph that uses it checks its record alone and places it there as a subgraph, whose
    nodes run where the node stands.

    It has no function, so calling it is a type error; its fields are the graph's, worked out at that load.

    Attributes:
        graph: The graph record this subgraph is.
    """

    graph: GraphRecord

    @property
    def version_chain(self) -> tuple[NodeDefinition, ...]:
        """This version, then each version it replaces, newest first."""
        return (self, *self._replaced)

    def version_of(self, version: int) -> NodeDefinition:
        """This node at ``version``, walking back through what each version replaces; a ``KeyError`` naming the versions
        there are when it has none. ``version in definition.versions`` asks first."""
        return _version_of(self.version_chain, version)

    def _where_declared(self) -> str:
        """Returns where this node was written, for a message that names it: a subgraph has no module, so its title
        and versions.

        For ``subgraph("loud", title="Loud lines")`` at its second version, it is ``the subgraph titled 'Loud lines'
        (versions 2, 1)``.
        """
        return f"the subgraph titled {self.title!r} (versions {', '.join(map(str, self.versions))})"

#: One version of a node: a function, or a subgraph.
type NodeDefinition = FunctionDefinition[..., Any] | SubgraphDefinition


def _version_of(version_chain: tuple[NodeDefinition, ...], version: int) -> NodeDefinition:
    for found in version_chain:
        if found.version == version:
            return found
    raise KeyError(f"{version_chain[0].id!r} has no version {version!r}; it has {', '.join(str(step.version) for step in version_chain)}")


class _NodeDescription(ConductorModel):
    """What an editor's palette shows of every node. Never made on its own: a node is described as a function or as a
    subgraph.

    Attributes:
        id: The id a graph names the node by.
        version: The version described, the newest.
        versions: Every version there is, newest first.
        title: What a person reads as its name.
        description: What it does, for a person who is no developer.
        category: The group a palette lists it under.
        tags: Words a palette filters by.
        extra: What the host stored on the node, which conductor never reads.
        docs: A longer text or a link about the node.
        deprecation: The notice that this version is going away, when it is.
        inputs: Its inputs.
        outputs: Its outputs.
    """

    id: str
    version: int
    versions: list[int]
    title: str
    description: str
    category: str
    tags: list[str]
    extra: dict[str, JsonValue]
    docs: str | None
    deprecation: Deprecation | None
    inputs: list[FieldDescription]
    outputs: list[FieldDescription]


class FunctionNodeDescription(_NodeDescription):
    """A node that is a function, as an editor's palette shows it.

    Attributes:
        kind: ``"function"``.
        policy: How the engine calls it.
        takes_added_inputs: Whether what is typed into it adds inputs.
        computes_outputs: Whether its outputs depend on what is typed into it and on what arrives.
        asks: Whether a call may wait for a person: its function raises ``Ask``, or it has an ``Asked`` parameter. A
            graph with such a node can pause, so a host can't run it unattended.
        asked: The ``Asked`` parameter a person may be asked to fill, drawn like an input though no edge reaches it;
            ``None`` when the node has none.
    """

    kind: Literal["function"] = "function"
    policy: Policy
    takes_added_inputs: bool
    computes_outputs: bool
    asks: bool
    asked: FieldDescription | None


class SubgraphDescription(_NodeDescription):
    """A subgraph as an editor's palette shows it.

    Its inputs and outputs are the graph inputs and graph outputs of its graph record, so only a registry that
    has the node definitions that record uses can describe it.

    Attributes:
        kind: ``"subgraph"``.
    """

    kind: Literal["subgraph"] = "subgraph"


#: A node as an editor's palette shows it, one of two kinds told apart by ``kind``.
NodeDescription = Annotated[FunctionNodeDescription | SubgraphDescription, PydanticField(discriminator="kind")]


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
      the call (``ask_not_declared``). A parameter a person may fill is
      ``Asked`` instead, and needs no ``asks``.

    A graph record becomes a node, a subgraph, with ``subgraph``.
    """
    number = _version_number("@node", id, title, description, version, replaces, upgrade)

    def make[**P, R](target: NodeFunction[P, R], /) -> FunctionDefinition[P, R]:
        try:
            if inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target):
                raise NodeDeclarationError(declaration.async_run())
            inputs, context, outputs, returns, asked = _read_signature(target, compute_outputs is not None)
            if asks and asked is not None:
                raise NodeDeclarationError(declaration.asked_and_asks(), asked.field.name)
            takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in inspect.signature(target).parameters.values())
            if compute_added_inputs is not None and not takes_kwargs:
                raise NodeDeclarationError(declaration.compute_added_inputs_without_var_keyword())
            if context is not None and policy is not None and policy.costs_nothing:
                raise NodeDeclarationError(declaration.costs_nothing_takes_context(), context.name)
            if compute_added_inputs is None and takes_kwargs:
                raise NodeDeclarationError(declaration.var_keyword_without_hook())
            hooks: dict[HookName, Callable[..., Any] | None] = {"compute_added_inputs": compute_added_inputs, "compute_outputs": compute_outputs}
            hook_parameters = {
                kind: _check_hook(hook, kind, inputs, adds_inputs=compute_added_inputs is not None)
                for kind, hook in hooks.items() if hook is not None
            }
            definition = FunctionDefinition(
                function=target, inputs=inputs, outputs=outputs, returns=returns, context=context,
                id=id, version=number, title=title, description=description,
                category=category, tags=tuple(tags), extra=MappingProxyType(dict(extra)), docs=docs, deprecation=deprecation,
                replaces=replaces, upgrade=upgrade or Upgrade(), policy=policy or Policy(),
                compute_added_inputs=compute_added_inputs, compute_outputs=compute_outputs, hook_parameters=hook_parameters,
                generic=any(field.whole.variables for field in inputs.values()), asks=asks, asked=asked,
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
    ``@node``. There is no ``policy`` and there are no hooks: the nodes inside
    the subgraph are called by their own. It registers nothing:
    ``registry.add(enrich)`` offers it, and ``registry.subgraph`` takes
    the same arguments and adds the node too.
    """
    number = _version_number("subgraph", id, title, description, version, replaces, upgrade)

    def make(record: GraphRecord | str | bytes, /) -> SubgraphDefinition:
        try:
            graph = read_record(record)
        except RefusedError as refused:
            raise InvalidCallError(refused.problems) from None
        return SubgraphDefinition(
            graph=graph, id=id, version=number, title=title, description=description, category=category,
            tags=tuple(tags), extra=MappingProxyType(dict(extra)), docs=docs, deprecation=deprecation, replaces=replaces,
            upgrade=upgrade or Upgrade(),
        )

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
        raise NodeDeclarationError(declaration.invalid_node_id(declared_with=declared_with, id=id))
    if not title.strip() or not description.strip():
        raise NodeDeclarationError(declaration.empty_title_or_description(declared_with=declared_with, id=id, empty="title" if not title.strip() else "description"))
    if upgrade is not None and replaces is None:
        raise NodeDeclarationError(declaration.upgrade_without_replaces(declared_with=declared_with, id=id))
    if replaces is not None and replaces.id != id:
        raise NodeDeclarationError(declaration.replaces_another_node(declared_with=declared_with, id=id, replaces=repr(replaces)))
    number = version if version is not None else (replaces.version + 1 if replaces is not None else 1)
    if replaces is not None and number <= replaces.version:
        raise NodeDeclarationError(declaration.version_not_after(declared_with=declared_with, id=id, version=number, replaced=replaces.version))
    return number


def _read_signature(
    function: Callable[..., Any], computed_outputs: bool,
) -> tuple[dict[str, Field], ContextParameter | None, dict[str, Field], ReturnShape, AskedParameter | None]:
    """Reads a function's signature.

    A type variable in it (``def keep[T](value: T) -> T``) stays in the field's hint, to be bound when a graph loads.

    Returns:
        The inputs by name, the ``Context`` parameter (``None`` when the first parameter is not one), the outputs
        by name, what kind of value the function returns them as, and the ``Asked`` parameter (``None`` for none).

    Raises:
        NodeDeclarationError: A parameter or the return type is declared in a way a node can't have.
    """
    parameters = inspect.signature(function).parameters  # read first: for what is not a function it raises Python's "is not a callable object"
    hints = typing.get_type_hints(function, include_extras=True)
    inputs: dict[str, Field] = {}
    context: ContextParameter | None = None
    asked: tuple[str, Any, Asked] | None = None
    for position, (name, parameter) in enumerate(parameters.items()):
        # A Context anywhere else is read as an input below, and its type refuses to be read (context_misplaced).
        if position == 0 and name in hints and is_context(hints[name]) and parameter.kind in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD):
            needs = typing.get_args(hints[name])[0]
            if not isinstance(needs, type):
                raise NodeDeclarationError(declaration.context_not_a_class(), name)
            context = ContextParameter(name, needs)
            continue
        if parameter.kind is parameter.VAR_KEYWORD:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.POSITIONAL_ONLY):
            raise NodeDeclarationError(declaration.parameter_not_by_name(), name)
        if name not in hints:
            raise NodeDeclarationError(declaration.input_without_type(), name)
        hint, marker = _without_asked(hints[name], name)
        if marker is not None:
            if asked is not None:
                raise NodeDeclarationError(declaration.asked_twice(), name)
            if parameter.default is not Parameter.empty:
                raise NodeDeclarationError(declaration.asked_with_default(), name)
            asked = (name, hint, marker)
            continue
        default = REQUIRED if parameter.default is Parameter.empty else parameter.default
        inputs[name] = declared_field(name, hint, default)
        if inputs[name].param.exclusive_group is not None:
            raise NodeDeclarationError(declaration.exclusive_group_on_input(), name)
    if "return" not in hints:
        raise NodeDeclarationError(declaration.output_without_type())
    returns = hints["return"]
    _refuse_asked(returns, None)
    outputs: dict[str, Field] = {}
    kind: ReturnShape
    outputs_class = typing.get_origin(returns) or returns  # a generic dataclass of outputs is written with its arguments
    if computed_outputs:
        kind = HookOutputs()
    elif returns is None or returns is type(None):
        kind = NoOutputs()
    elif isinstance(outputs_class, type) and dataclasses.is_dataclass(outputs_class):
        if not outputs_class.__dataclass_params__.frozen:  # type: ignore[attr-defined]
            raise NodeDeclarationError(declaration.outputs_not_frozen())
        given = dict(zip(getattr(outputs_class, "__parameters__", ()), typing.get_args(returns), strict=True))  # the class's variables as the signature names them
        own = typing.get_type_hints(outputs_class, include_extras=True)
        for name, hint in own.items():
            _refuse_asked(hint, name)
        outputs = {f.name: declared_field(f.name, Type(own[f.name]).bind(given).hint if given else own[f.name], _declared_default(f)) for f in dataclasses.fields(outputs_class)}
        kind = OutputsClass(outputs_class)
    else:
        outputs, kind = {"result": Field("result", returns)}, OneResult()
    on_inputs = {variable for field in inputs.values() for variable in field.whole.variables}
    for field in (*inputs.values(), *outputs.values()):
        for variable in field.whole.variables:
            if variable.__constraints__:
                raise NodeDeclarationError(declaration.type_variable_constrained(variable=variable.__name__), field.name)
            if variable not in on_inputs:
                raise NodeDeclarationError(declaration.type_variable_only_in_return(variable=variable.__name__), field.name)
    if asked is None:
        return inputs, context, outputs, kind, None
    name, hint, marker = asked
    return inputs, context, outputs, kind, _asked_parameter(function, declared_field(name, hint, REQUIRED), marker, inputs, context)


def _without_asked(hint: Any, name: str) -> tuple[Any, Asked | None]:
    """A parameter's hint with the ``Asked`` around it taken off, and that ``Asked``.

    ``Annotated[bool, Param(title="OK"), Asked(approval)]`` is ``Annotated[bool, Param(title="OK")]`` and
    ``Asked(approval)``; a hint with no ``Asked`` around it is itself and ``None``.

    Raises:
        NodeDeclarationError: Two ``Asked`` around one parameter (``asked_twice``), or one inside its type
            (``asked_not_on_a_parameter``).
    """
    marker: Asked | None = None
    if typing.get_origin(hint) is Annotated:
        markers = [m for m in hint.__metadata__ if isinstance(m, Asked)]
        if len(markers) > 1:
            raise NodeDeclarationError(declaration.asked_twice(), name)
        if markers:
            (marker,) = markers
            kept = tuple(m for m in hint.__metadata__ if not isinstance(m, Asked))
            hint = Annotated[(hint.__origin__, *kept)] if kept else hint.__origin__
    _refuse_asked(hint, name)
    return hint, marker


def _refuse_asked(hint: Any, name: str | None) -> None:
    """Raises when ``hint`` holds an ``Asked``: the hint of an output, or a parameter's with the ``Asked`` around it
    taken off.

    Raises:
        NodeDeclarationError: ``asked_not_on_a_parameter``, naming the field when there is a name.
    """
    if any(isinstance(m, Asked) for m in split_hint(hint)[2]):
        raise NodeDeclarationError(declaration.asked_not_on_a_parameter(), *([name] if name else []))


def _asked_parameter(
    function: Callable[..., Any], field: Field, marker: Asked, inputs: Mapping[str, Field], context: ContextParameter | None,
) -> AskedParameter:
    """Reads an ``Asked`` parameter's function against the node it is on.

    For ``ok: Annotated[bool, Asked(approval)]`` on ``approve(invoice: Invoice, ok: ...)``, with
    ``approval(invoice: Invoice) -> bool``, it is ``ok``, filled by ``approval``, which takes the input ``invoice``.

    Args:
        function: The node's function, whose signature names the parameter's type.
        field: The parameter, read by its hint without the ``Asked``.
        marker: The ``Asked``.
        inputs: The node's inputs, by name.
        context: The node's ``Context`` parameter, or ``None``.

    Raises:
        NodeDeclarationError: Naming the parameter: the function is ``async def`` (``async_run``), one of its
            parameters is not a parameter of the node taken by name (``asking_function_parameter_unknown``), or it
            does not return the parameter's type (``asking_function_returns_another_type``).
    """
    asking = marker.function
    name = field.name
    if inspect.iscoroutinefunction(asking) or inspect.isasyncgenfunction(asking):
        raise NodeDeclarationError(declaration.async_run(), name)
    may_name = [*inputs, *([context.name] if context is not None else [])]
    taken: list[str] = []
    for parameter in inspect.signature(asking).parameters.values():
        if parameter.name not in may_name or parameter.kind not in (parameter.POSITIONAL_OR_KEYWORD, parameter.KEYWORD_ONLY):
            raise NodeDeclarationError(declaration.asking_function_parameter_unknown(parameter=parameter.name, parameters=may_name), name)
        taken.append(parameter.name)
    returns = typing.get_type_hints(asking).get("return", Parameter.empty)
    declared = typing.get_type_hints(function)[name]
    if returns != declared:
        written = "nothing declared" if returns is Parameter.empty else Type(returns).id
        raise NodeDeclarationError(declaration.asking_function_returns_another_type(returns=written, type=field.whole.id), name)
    context_name = context.name if context is not None and context.name in taken else None
    return AskedParameter(field, asking, tuple(n for n in taken if n != context_name), context_name, AskedValueDoor(field))


def _declared_default(declared: DataclassField[Any]) -> Any:
    """A dataclass field's default, which a person asked for the output sees proposed; ``REQUIRED`` when it has none."""
    if declared.default is not dataclasses.MISSING:
        return declared.default
    if declared.default_factory is not dataclasses.MISSING:
        return declared.default_factory()
    return REQUIRED


def declared_inputs(definition: NodeDefinition) -> Mapping[str, Field]:
    """The inputs a version declares; a subgraph declares none of its own, its fields being known only when a graph using it loads."""
    match definition:
        case FunctionDefinition():
            return definition.inputs
        case SubgraphDefinition():
            return {}


def read_by_hooks(definition: FunctionDefinition[..., Any]) -> frozenset[str]:
    """The inputs a node's hooks read as values: fixed when the graph loads, so never open to a caller."""
    return frozenset(name for parameters in definition.hook_parameters.values() for name, receives in parameters.items() if receives == "value")


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
            raise NodeDeclarationError(declaration.arriving_on_compute_added_inputs(parameter=name))
        if parameter.kind is parameter.VAR_KEYWORD:
            if not (arriving and adds_inputs):
                raise NodeDeclarationError(declaration.hook_parameter_var_keyword(hook=kind, parameter=name))
            parameters[name] = "added_arriving_types"
        elif parameter.kind is not parameter.POSITIONAL_OR_KEYWORD and parameter.kind is not parameter.KEYWORD_ONLY:
            raise NodeDeclarationError(declaration.hook_parameter_not_by_name(hook=kind, parameter=name))
        elif arriving:
            parameters[name] = "arriving_type"
        elif name not in inputs:
            raise NodeDeclarationError(declaration.hook_parameter_unknown(hook=kind, parameter=name, inputs=sorted(inputs)))
        else:
            parameters[name] = "value"
    return parameters


def _raise_if_upgrade_is_missing(older: NodeDefinition, newer: FunctionDefinition[..., Any]) -> None:
    """A field name of the replaced version that is gone from the newer one needs an upgrade to say where it went.

    A gone input is explained by a rename or a ``rewrite``, which rewrites what is typed in; a gone output only by a
    rename, since the edges out of it are what moves.
    """
    match older:
        case SubgraphDefinition():  # its fields are known only when a graph using it loads
            return
        case FunctionDefinition():
            gone_inputs = set(older.inputs) - set(newer.inputs)
            gone_outputs = set(older.outputs) - set(newer.outputs)
    upgrade = newer.upgrade
    unexplained = gone_outputs - set(upgrade.renamed_outputs)
    if upgrade.rewrite is None:
        unexplained |= gone_inputs - set(upgrade.renamed_inputs)
    if unexplained:
        raise NodeDeclarationError(declaration.upgrade_missing(gone=sorted(unexplained)))
