"""Node definitions: what ``@node`` and ``subgraph()`` make.

A node definition is one version of a node. It comes in two kinds. A ``FunctionDefinition`` wraps a Python
function, and a ``SubgraphDefinition`` wraps a graph record. ``NodeDefinition`` is either one.

For example, ``@node("shout", ...)`` on ``def shout(line: str, mark: str = "!") -> str`` makes a
``FunctionDefinition`` with the inputs ``line`` and ``mark`` and the output ``result``. ``declare.py`` builds it
from the function's signature.

A newer version names the one it replaces, and an ``Upgrade`` says how a graph moves from one to the other.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, Self

from pydantic import JsonValue

from conductor._internal.base import REQUIRED
from conductor._internal.context import ContextParameter
from conductor._internal.field import Field, FieldDescription
from conductor._internal.record import GraphRecord
from conductor._internal.utils import qualified_name

from .describe import NodeDescription
from .policy import Deprecation, Policy
from .running_calls import RunningCalls

if TYPE_CHECKING:
    from .hook import Hook


class NodeFunction[**P, R](Protocol):
    """The function a node is made from: it can be called, and it knows where it was written.

    Every function, method and static method has these names. A message about the node uses them to say where the
    node was declared.
    """

    __name__: str
    __module__: str
    __qualname__: str

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...


#: The names of the hooks a node may have, the same as the keywords ``@node`` takes them by.
HookName = Literal["compute_added_inputs", "compute_outputs"]


@dataclass(frozen=True)
class Upgrade:
    """How a graph that uses an older version of a node moves to the newer one.

    It goes on the newer version's ``@node``, beside ``replaces``. It is needed only when the newer version drops
    an input or an output the older one had. A dropped input needs a rename or a ``rewrite``, and a dropped output
    needs a rename.

    Only ``registry.upgrade`` uses it. Compiling never upgrades, so a graph saved at an old version runs that
    version.

    Attributes:
        renamed_inputs: Old input names to new ones. Each input's constant, default and edges move to the new
            name, before ``rewrite`` runs.
        renamed_outputs: Old output names to new ones. Every edge that reads the old output, or a field of it,
            moves to the new name.
        rewrite: A function that takes the node's constants and defaults as one dict, and returns them for the
            newer version. A key that was a default stays a default, and any other key becomes a constant.
    """

    renamed_inputs: Mapping[str, str] = dataclasses.field(default_factory=dict[str, str])
    renamed_outputs: Mapping[str, str] = dataclasses.field(default_factory=dict[str, str])
    rewrite: Callable[[dict[str, JsonValue]], Mapping[str, Any]] | None = None


@dataclass(frozen=True, eq=False, kw_only=True)
class _Version:
    """What every version of a node has, whether it is a function or a subgraph.

    It is never made on its own: a version is a ``FunctionDefinition`` or a ``SubgraphDefinition``. Each of them
    answers ``where_declared``, ``input_names``, ``required_inputs`` and ``output_names`` in its own way. So a
    caller never has to ask which kind it has.

    Attributes:
        id: The id a graph names the node by. Every saved graph that uses the node stores it, so changing it means
            changing saved data, not just a rename.
        version: Which version of the node this is, counted from 1. A graph names the version it uses.
        title: The node's name, as a person reads it.
        description: What the node does, in a sentence for a person. Never the docstring.
        category: The group an editor lists the node under. The host decides what the groups are.
        tags: The host's labels for finding the node.
        extra: The host's own JSON about the node. Conductor keeps it and shows it in ``describe()``, but never
            looks at it.
        docs: Longer markdown help, for when the description isn't enough.
        deprecation: Set when this version is going away: what to tell a person, and what to use instead.
        replaces: The version before this one, with the same id, or ``None`` for a first version. It may be of
            the other kind.
        upgrade: How a graph moves from ``replaces`` to this version. Used only when ``replaces`` is set.
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
    def versions(self) -> Mapping[int, Self | NodeDefinition]:
        """Every version of this node, by version number, newest first.

        If ``v3`` replaces ``v2``, which replaces ``v1``, then ``v3.versions`` is ``{3: v3, 2: v2, 1: v1}``. The
        mapping is read-only, since the definition is frozen.
        """
        older = {} if self.replaces is None else self.replaces.versions
        return MappingProxyType({self.version: self, **older})

    def described_as(
        self,
        *,
        inputs: list[FieldDescription],
        outputs: list[FieldDescription],
        asks: bool,
        takes_added_inputs: bool,
        computes_outputs: bool,
    ) -> NodeDescription:
        """Describes this version for an editor's palette.

        The parts that differ between a function and a subgraph are passed in, since each kind works them out its
        own way.

        Args:
            inputs: The node's inputs.
            outputs: The node's outputs.
            asks: Whether a call may wait for a person.
            takes_added_inputs: Whether the node's constants can add inputs.
            computes_outputs: Whether the node's outputs depend on its constants and on the types its edges bring.
        """
        return NodeDescription(
            id=self.id,
            version=self.version,
            versions=list(self.versions),
            title=self.title,
            description=self.description,
            category=self.category,
            tags=list(self.tags),
            extra=dict(self.extra),
            docs=self.docs,
            deprecation=self.deprecation,
            inputs=inputs,
            outputs=outputs,
            asks=asks,
            takes_added_inputs=takes_added_inputs,
            computes_outputs=computes_outputs,
        )


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)  # the version's own __repr__
class FunctionDefinition[**P, R](_Version):
    """A version of a node that is a Python function.

    ``@node`` on a ``def`` makes one. It is frozen, and it can be called with the function's own signature, so a
    node is tested by calling it. A host gets it from ``registry[id]`` or by looping over a registry.

    Attributes:
        function: The function each call runs.
        inputs: Each input by name, as the signature declares it. The ``Context`` parameter isn't an input, it is
            ``context``. An input a hook adds isn't here either. It exists only on the node in a compiled graph.
        outputs: Each output by name: ``result`` for one value, or one per field of a returned frozen dataclass
            with no ``@host_type``.
        outputs_dataclass: The frozen dataclass the function returns its outputs in. ``None`` when it returns one
            value (a dataclass declared with ``@host_type`` is one), a mapping from its ``compute_outputs`` hook,
            or nothing.
        policy: How the engine calls the node: how many calls at once, retries, a time limit, and whether it
            reuses earlier results.
        compute_added_inputs: The hook that adds inputs from the node's constants, when a graph is compiled. The
            function takes them as ``**kwargs``.
        compute_outputs: The hook that gives every output, from the node's constants and the types its edges bring,
            when a graph is compiled. The function returns them as a mapping.
        context: The first parameter, when it is ``Context[X]``. Each call gets the run's context there, first by
            position. ``None`` for a node that takes no context.
        asks: Whether the function may raise ``Ask`` to wait for a person, as ``@node(asks=True)`` says. The answer
            then becomes its outputs. Without it, ``@node`` or the run refuses the ``Ask``
            (``python -m conductor.about asking-a-person``).
        running_calls: The counter of this node's running calls across the process, which ``policy.concurrency``
            limits. ``@node`` makes it, and every copy a registry makes of the definition shares it.
        code_text: The function's code as text that's the same on every Python version, without its docstrings
            (see ``read_code`` in ``source.py``). ``@node`` works it out when it declares the function, so a
            file replaced on disk later, as a deploy does while old processes still run, doesn't change it. A
            compiled node hashes it into its code fingerprint, which says whether an earlier call can be reused.
    """

    function: NodeFunction[P, R]
    inputs: Mapping[str, Field]
    outputs: Mapping[str, Field]
    outputs_dataclass: type | None
    code_text: str
    policy: Policy = Policy()
    compute_added_inputs: Hook | None = None
    compute_outputs: Hook | None = None
    context: ContextParameter | None = None
    asks: bool = False
    running_calls: RunningCalls = dataclasses.field(default_factory=RunningCalls)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Calls the function, as a test or a host would."""
        return self.function(*args, **kwargs)

    @property
    def where_declared(self) -> str:
        """Where the function was written, for messages about the node.

        ``def shout`` in the module ``readme`` gives ``readme.shout``.
        """
        return qualified_name(self.function)

    @property
    def input_names(self) -> tuple[str, ...]:
        """The names of the inputs its signature declares."""
        return tuple(self.inputs)

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """The names of the inputs an edge or a graph input must give a value.

        That is each input with no default that isn't ``constant_only``.
        """
        return tuple(
            name
            for name, field in self.inputs.items()
            if field.default is REQUIRED and not field.param.constant_only
        )

    @property
    def output_names(self) -> tuple[str, ...]:
        """The names of the outputs."""
        return tuple(self.outputs)

    def describe(self) -> NodeDescription:
        """Describes the node for an editor's palette."""
        return self.described_as(
            inputs=[field.description for field in self.inputs.values()],
            outputs=[field.description for field in self.outputs.values()],
            takes_added_inputs=self.compute_added_inputs is not None,
            computes_outputs=self.compute_outputs is not None,
            asks=self.asks,
        )


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)  # the version's own __repr__
class SubgraphDefinition(_Version):
    """A version of a node that is a whole graph, used as one step in another graph.

    ``subgraph(...)(record)`` makes one. When a graph that uses it is compiled, the subgraph's own record is
    compiled on its own first. Its nodes then run in the outer graph where the subgraph node is.

    It has no function, so calling it is a type error. Its inputs and outputs are the graph's, and they are worked
    out when it is compiled.

    Attributes:
        graph: The graph record this subgraph is.
    """

    graph: GraphRecord

    @property
    def where_declared(self) -> str:
        """The subgraph's title and versions, for messages about the node.

        A subgraph has no module, so this gives something like ``the subgraph titled 'Loud lines' (versions 2, 1)``.
        """
        return f"the subgraph titled {self.title!r} (versions {', '.join(map(str, self.versions))})"

    @property
    def input_names(self) -> tuple[str, ...]:
        """The names of the graph inputs in its record, sorted."""
        return self.graph.input_names

    @property
    def required_inputs(self) -> tuple[str, ...]:
        """The names of the graph inputs in its record that have no default, sorted."""
        return self.graph.required_inputs

    @property
    def output_names(self) -> tuple[str, ...]:
        """The names of the graph outputs in its record, sorted."""
        return tuple(sorted(self.graph.outputs))


#: One version of a node: a function, or a subgraph.
type NodeDefinition = FunctionDefinition[..., Any] | SubgraphDefinition


def gone_inputs(older: NodeDefinition, newer: NodeDefinition) -> set[str]:
    """Returns the inputs of ``older`` that ``newer``, the version replacing it, neither has nor renames."""
    return set(older.input_names) - set(newer.input_names) - newer.upgrade.renamed_inputs.keys()


def gone_outputs(older: NodeDefinition, newer: NodeDefinition) -> set[str]:
    """Returns the outputs of ``older`` that ``newer``, the version replacing it, neither has nor renames."""
    return set(older.output_names) - set(newer.output_names) - newer.upgrade.renamed_outputs.keys()


def with_defaults(
    definition: NodeDefinition,
    *,
    tags: tuple[str, ...],
    category: str,
    extra: Mapping[str, JsonValue],
) -> NodeDefinition:
    """Returns ``definition`` with a registry's default tags, category and ``extra`` filled in.

    The node's own values win. Say ``shout`` has ``tags=("text",)`` and no category, and it is added to a registry
    with ``tags=("demo",)`` and ``category="Text"``. It gets ``tags=("demo", "text")``, with the registry's tags
    first, and ``category="Text"``. A node's own category and its own ``extra`` keys are kept.

    Every older version it replaces gets the same defaults. If nothing changes, ``definition`` itself is returned.
    """
    merged_tags = tuple(dict.fromkeys((*tags, *definition.tags)))
    merged_category = definition.category or category
    merged_extra = {**extra, **definition.extra}
    replaces = (
        None
        if definition.replaces is None
        else with_defaults(definition.replaces, tags=tags, category=category, extra=extra)
    )
    if (merged_tags, merged_category, merged_extra, replaces) == (
        definition.tags,
        definition.category,
        dict(definition.extra),
        definition.replaces,
    ):
        return definition
    return dataclasses.replace(
        definition,
        tags=merged_tags,
        category=merged_category,
        extra=MappingProxyType(merged_extra),
        replaces=replaces,
    )
