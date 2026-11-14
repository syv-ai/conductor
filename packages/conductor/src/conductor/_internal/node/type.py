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
from types import MappingProxyType
from typing import Any, Literal, Protocol, Self

from pydantic import JsonValue

from conductor._internal.base import REQUIRED, qualified_name
from conductor._internal.context import ContextParameter
from conductor._internal.field import Field, FieldDescription
from conductor._internal.record import GraphRecord

from .describe import NodeDescription
from .policy import Deprecation, Policy
from .running_calls import RunningCalls


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


@dataclass(frozen=True, eq=False, kw_only=True)
class _Version:
    """What every version of a node has, whether it is a function or a subgraph.

    Never made on its own: a version is a ``FunctionDefinition`` or a ``SubgraphDefinition``, the two kinds of
    ``NodeDefinition``. Each kind answers ``where_declared``, ``input_names``, ``required_inputs`` and
    ``output_names`` in its own way, so no caller asks which kind it holds to learn them.

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
    def version_chain(self) -> tuple[Self | NodeDefinition, ...]:
        """This version, then each version it replaces, newest first."""
        return (self,) if self.replaces is None else (self, *self.replaces.version_chain)

    @functools.cached_property
    def versions(self) -> tuple[int, ...]:
        """Every version number, newest first."""
        return tuple(step.version for step in self.version_chain)

    def version_of(self, version: int) -> Self | NodeDefinition:
        """Returns this node at ``version``, walking back through what each version replaces.

        Raises:
            KeyError: It has no such version, naming the versions it has; ``version in definition.versions`` asks first.
        """
        for found in self.version_chain:
            if found.version == version:
                return found
        raise KeyError(f"{self.id!r} has no version {version!r}; it has {', '.join(map(str, self.versions))}")

    def described_as(
        self, *, inputs: list[FieldDescription], outputs: list[FieldDescription], asks: bool, takes_added_inputs: bool, computes_outputs: bool,
    ) -> NodeDescription:
        """Describes this version as an editor's palette shows it, with what its kind of node works out itself.

        Args:
            inputs: Its inputs.
            outputs: Its outputs.
            asks: Whether a call may wait for a person.
            takes_added_inputs: Whether what is typed into it adds inputs.
            computes_outputs: Whether its outputs depend on what is typed into it and on what arrives.
        """
        return NodeDescription(
            id=self.id, version=self.version, versions=list(self.versions), title=self.title, description=self.description,
            category=self.category, tags=list(self.tags), extra=dict(self.extra), docs=self.docs, deprecation=self.deprecation,
            inputs=inputs, outputs=outputs, asks=asks, takes_added_inputs=takes_added_inputs, computes_outputs=computes_outputs,
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
        outputs_dataclass: The frozen dataclass the function returns its outputs as, one output per field. ``None``
            when it returns one value (``result``), the mapping of outputs its ``compute_outputs`` hook gave, or nothing.
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
        asks: Whether a call may wait for a person: the function may raise ``Ask``, as ``@node(asks=True)``
            declares, and the answer is then its outputs. An ``Ask`` raised from any other node fails its call
            (``ask_not_declared``).
        running_calls: How many calls of it are running now, in every run of the process, which ``policy``'s
            ``concurrency`` caps when it sets one. Made by ``@node`` and shared by every copy of the definition a registry makes.
    """

    function: NodeFunction[P, R]
    inputs: Mapping[str, Field]
    outputs: Mapping[str, Field]
    outputs_dataclass: type | None
    policy: Policy = Policy()
    compute_added_inputs: Callable[..., Mapping[Any, Any]] | None = None
    compute_outputs: Callable[..., Mapping[Any, Any]] | None = None
    context: ContextParameter | None = None
    hook_parameters: Mapping[HookName, Mapping[str, HookParameter]] = dataclasses.field(default_factory=dict[HookName, Mapping[str, HookParameter]])
    asks: bool = False
    running_calls: RunningCalls = dataclasses.field(default_factory=RunningCalls)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the function, as a test or a host would."""
        return self.function(*args, **kwargs)

    @property
    def where_declared(self) -> str:
        """Where the function was written, for a message that names the node: ``readme.shout`` for ``def shout`` in
        the module ``readme``."""
        return qualified_name(self.function)

    @property
    def input_names(self) -> tuple[str, ...]:
        """The names of the inputs its signature declares."""
        return tuple(self.inputs)

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """The inputs a graph must give a value to: those with no default that no hook reads."""
        hooked = read_by_hooks(self)
        return tuple(name for name, field in self.inputs.items() if field.default is REQUIRED and name not in hooked)

    @property
    def output_names(self) -> tuple[str, ...]:
        """The names of its outputs."""
        return tuple(self.outputs)

    def describe(self) -> NodeDescription:
        """The node as an editor's palette shows it."""
        return self.described_as(
            inputs=[field.description for field in self.inputs.values()], outputs=[field.description for field in self.outputs.values()],
            takes_added_inputs=self.compute_added_inputs is not None, computes_outputs=self.compute_outputs is not None,
            asks=self.asks,
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
    def where_declared(self) -> str:
        """Its title and versions, for a message that names the node, as a subgraph has no module: ``the subgraph
        titled 'Loud lines' (versions 2, 1)``."""
        return f"the subgraph titled {self.title!r} (versions {', '.join(map(str, self.versions))})"

    @property
    def input_names(self) -> tuple[str, ...]:
        """The graph inputs its record names."""
        return self.graph.input_names

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """The graph inputs its record gives no default."""
        return self.graph.required_inputs

    @property
    def output_names(self) -> tuple[str, ...]:
        """The graph outputs its record declares."""
        return tuple(self.graph.outputs)


#: One version of a node: a function, or a subgraph.
type NodeDefinition = FunctionDefinition[..., Any] | SubgraphDefinition


def gone_inputs(older: NodeDefinition, newer: NodeDefinition) -> set[str]:
    """The inputs of ``older`` that ``newer``, the version replacing it, neither has nor renames."""
    return set(older.input_names) - set(newer.input_names) - newer.upgrade.renamed_inputs.keys()


def gone_outputs(older: NodeDefinition, newer: NodeDefinition) -> set[str]:
    """The outputs of ``older`` that ``newer``, the version replacing it, neither has nor renames."""
    return set(older.output_names) - set(newer.output_names) - newer.upgrade.renamed_outputs.keys()


def with_defaults(definition: NodeDefinition, *, tags: tuple[str, ...], category: str, extra: Mapping[str, JsonValue]) -> NodeDefinition:
    """Returns ``definition``, and each version it replaces, with a registry's defaults under its own.

    ``shout`` declared with ``tags=("text",)`` and no category, added to a registry with ``tags=("demo",)`` and
    ``category="Text"``, becomes ``tags=("demo", "text")`` and ``category="Text"``. The registry's tags come first;
    the node's own category, and its own ``extra`` keys, win. When nothing changes, it is ``definition`` itself.
    """
    merged_tags = tuple(dict.fromkeys((*tags, *definition.tags)))
    merged_category = definition.category or category
    merged_extra = {**extra, **definition.extra}
    replaces = None if definition.replaces is None else with_defaults(definition.replaces, tags=tags, category=category, extra=extra)
    if (merged_tags, merged_category, merged_extra, replaces) == (definition.tags, definition.category, dict(definition.extra), definition.replaces):
        return definition
    return dataclasses.replace(definition, tags=merged_tags, category=merged_category, extra=MappingProxyType(merged_extra), replaces=replaces)


def read_by_hooks(definition: FunctionDefinition[..., Any]) -> frozenset[str]:
    """The inputs a node's hooks read as values: fixed when the graph loads, so never open to a caller."""
    return frozenset(name for parameters in definition.hook_parameters.values() for name, receives in parameters.items() if receives == "value")
