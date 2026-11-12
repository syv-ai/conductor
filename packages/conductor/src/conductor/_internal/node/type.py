"""``FunctionDefinition`` and ``SubgraphDefinition``, the two kinds of ``NodeDefinition``: one version of a node.

``@node("shout", title="Shout", description="...")`` on ``def shout(line: str, mark: str = "!") -> str`` makes a
``FunctionDefinition`` with the inputs ``line`` and ``mark`` and the output ``result``; ``subgraph("loud", ...)(record)``
makes a ``SubgraphDefinition`` of a graph record. ``declare.py`` reads the function's signature into one.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import JsonValue

from conductor._internal.ask import Ask
from conductor._internal.base import qualified_name
from conductor._internal.context import ContextParameter
from conductor._internal.field import Field
from conductor._internal.gate import AskedValueDoor, RefusedError, ReturnShape
from conductor._internal.node.describe import FunctionNodeDescription, SubgraphDescription
from conductor._internal.node.policy import Deprecation, Policy
from conductor._internal.problem import catalogue as problem
from conductor._internal.record import GraphRecord


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

    def filling(self, node_function: Callable[..., Any]) -> Callable[..., Any]:
        """``node_function`` with this parameter filled: what a run calls for the node, once, when the graph is compiled.

        Called as the node's function is (its context first, when it takes one, then its arguments by name), it
        calls ``approval`` with the inputs it names unless a person's answer for ``ok`` is already among the
        arguments, reads what it returns by the door, then calls ``approve`` with ``ok``. An ``Ask`` from
        ``approval`` goes through; one from ``approve`` itself raises ``RefusedError`` with ``ask_not_declared``,
        since a node with an ``Asked`` parameter asks through it.
        """
        name, asking, inputs, context_name, door = self.field.name, self.function, self.inputs, self.context, self.door

        def filled(*context: Any, **arguments: Any) -> Any:
            if name not in arguments:
                given = {input_name: arguments[input_name] for input_name in inputs}
                if context_name is not None:
                    given[context_name] = context[0]
                arguments = {**arguments, name: door(asking(**given))}
            try:
                return node_function(*context, **arguments)
            except Ask:
                raise RefusedError([problem.ask_not_declared()]) from None

        return filled


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
