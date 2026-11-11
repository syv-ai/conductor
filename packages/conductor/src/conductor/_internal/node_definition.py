"""``@node``: a plain function, or a stored graph, becomes a frozen ``NodeDefinition``.

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
a frozen dataclass returned gives one output per field. ``Series[X]`` and ``Table[R]`` give many rows, ``SKIPPED`` on an output
is a branch not taken, and ``raise Ask(...)`` stops the row for a person.

A newer version of a node names the one it ``replaces``, itself a ``@node``
of the same id, and says with ``upgrade`` how a node saved at that one moves
here. A definition is one version, and each version names the one before it in
``replaces``::

    @node("split", title="Split", description="Splits the text into lines.")
    def split_v1(text: str, sep: str = "\\n") -> Series[str]: ...

    @node("split", title="Split", description="Splits the text into lines.", replaces=split_v1, upgrade=Upgrade(renamed_inputs={"sep": "separator"}))
    def split(text: str, separator: str = "\\n") -> Series[str]: ...

Two hooks shape a node from what it is given in one graph, each a plain
function run when a graph is loaded, never per row. A parameter of a hook
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

A stored graph is a node too: ``node("enrich", title="Enrich", description="...")(record)``. Loading a graph that
uses it puts the stored graph's nodes in its place.
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import math
import typing
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, Literal, ParamSpec, Protocol, TypeVar, overload

from pydantic import JsonValue

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import InvalidCallError, NodeDeclarationError, Problem, make_problem
from conductor._internal.fields import (
    REQUIRED,
    FieldDescription,
    FieldSpec,
    FieldValueError,
    _FromRunMarker,
    marked,
    read_field,
)
from conductor._internal.record import GraphRecord, as_record

_P = ParamSpec("_P")
_R = TypeVar("_R")
_R_co = TypeVar("_R_co", covariant=True)


class NodeFunction(Protocol[_P, _R_co]):
    """A function a node is made of: called with its own signature, and named where it was written.

    Every function, method and static method has these names, so a node says where it was declared from them.
    """

    __name__: str
    __module__: str
    __qualname__: str

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ...

#: The hooks a node may have, by the keyword ``@node`` takes each as.
HookName = Literal["compute_added_inputs", "compute_outputs"]

#: What a hook's parameter receives: the ``value`` typed into the input it names; the ``arriving_type`` on that input;
#: or, as ``**name: type``, the ``added_arriving_types``, the type arriving on each input ``compute_added_inputs`` added.
HookParameter = Literal["value", "arriving_type", "added_arriving_types"]


@dataclass(frozen=True)
class Policy:
    """How the engine calls a node, set by its author: ``@node("fetch", policy=Policy(retries=3))``.

    ``concurrency`` rows are called at once, on threads; with 1 and no
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
    world, not only its inputs: ``run.again`` calls it on every row.
    """

    concurrency: int = 1
    retries: int = 0
    retry_delay: float = 1.0
    attempt_timeout: float | None = None
    retry_on: tuple[type[BaseException], ...] = ()
    reuse_when_inputs_match: bool = True
    #: Whether a call goes to a thread: several at once, or a clock on each attempt.
    threaded: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        counted = self.concurrency >= 1 and self.retries >= 0
        timed = math.isfinite(self.retry_delay) and self.retry_delay >= 0 and (
            self.attempt_timeout is None or (math.isfinite(self.attempt_timeout) and self.attempt_timeout > 0)
        )
        if not (counted and timed):
            raise NodeDeclarationError(
                f"{self}: concurrency is at least 1, retries at least 0, retry_delay finite seconds of at least 0, attempt_timeout finite seconds above 0",
                code="policy_invalid",
            )
        object.__setattr__(self, "threaded", self.concurrency > 1 or self.attempt_timeout is not None)

    def __repr__(self) -> str:
        """The call that makes it, with only what differs from the defaults: ``Policy(retries=2)``."""
        fields = ("concurrency", "retries", "retry_delay", "attempt_timeout", "retry_on", "reuse_when_inputs_match")
        changed = [f"{name}={getattr(self, name)!r}" for name in fields if getattr(self, name) != getattr(_DEFAULT_POLICY, name)]
        return f"Policy({', '.join(changed)})"

    def describe(self) -> PolicyDescription:
        """The policy as an editor reads it; ``retry_on`` is classes, which have no JSON form, so it is left out."""
        return PolicyDescription(
            concurrency=self.concurrency, retries=self.retries, retry_delay=self.retry_delay,
            attempt_timeout=self.attempt_timeout, reuse_when_inputs_match=self.reuse_when_inputs_match,
        )


class PolicyDescription(ConductorModel):
    """A node's ``Policy`` as an editor reads it, ``retry_on`` left out."""

    concurrency: int
    retries: int
    retry_delay: float
    attempt_timeout: float | None
    reuse_when_inputs_match: bool


_DEFAULT_POLICY = Policy()


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

    renamed_inputs: Mapping[str, str] = field(default_factory=dict[str, str])
    renamed_outputs: Mapping[str, str] = field(default_factory=dict[str, str])
    rewrite: Callable[[dict[str, JsonValue]], Mapping[str, Any]] | None = None


@dataclass(frozen=True, eq=False, kw_only=True)
class _Version:
    """What every version of a node has, whether it is a function or a stored graph.

    Never made on its own: a version is a ``FunctionDefinition`` or a ``GraphDefinition``, the two kinds of
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
    extra: Mapping[str, JsonValue] = field(default_factory=dict[str, JsonValue])
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
        return () if self.replaces is None else self.replaces.lineage

    @property
    def versions(self) -> tuple[int, ...]:
        """Every version number, newest first."""
        return (self.version, *(step.version for step in self._replaced))

    def _described(self, *, is_graph: bool, policy: Policy, inputs: Iterable[FieldSpec], outputs: Iterable[FieldSpec], hooks: tuple[bool, bool]) -> NodeDescription:
        return NodeDescription(
            id=self.id, version=self.version, versions=list(self.versions), title=self.title, description=self.description,
            category=self.category, tags=list(self.tags), extra=dict(self.extra), docs=self.docs, deprecation=self.deprecation,
            is_graph=is_graph, policy=policy.describe(),
            inputs=[f.describe() for f in inputs], outputs=[f.describe() for f in outputs],
            takes_added_inputs=hooks[0], computes_outputs=hooks[1],
        )


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)  # the version's own __repr__
class FunctionDefinition(_Version, Generic[_P, _R]):
    """A version of a node that is a function: what it takes, what it gives, and how to call it.

    Frozen, and callable with the function's own signature, so a node is tested by calling it. ``@node`` on a
    ``def`` makes one; a host reads it from ``registry[id]`` or by looping over a registry.

    Attributes:
        function: What runs on each row.
        inputs: Each input by name, as its signature declares it. A ``FromRun`` parameter is not an input: it is in
            ``from_run``. An input a hook adds is not here either; a loaded graph's node holds it.
        outputs: Each output by name: ``result`` for one value, a field each for a frozen dataclass returned.
        returns: What the function returns: one ``result``, a frozen dataclass of outputs, the mapping of outputs a
            ``compute_outputs`` hook gave, or nothing.
        policy: How it runs: how many rows at once, its retries and time limit, whether it reuses earlier work.
        compute_added_inputs: A hook run at load that adds inputs from the values typed into it; the function takes
            them as ``**kwargs``.
        compute_outputs: A hook run at load that gives every output, from the values typed in and the types
            arriving; the function returns them as a mapping.
        from_run: The ``FromRun`` parameters: the type whose value goes under each name in a call.
        hook_parameters: Each hook's parameters, by the hook's name (``compute_added_inputs``, ``compute_outputs``)
            and then the parameter's: what the parameter receives. Read from the hooks' signatures once, at
            ``@node``.
    """

    function: NodeFunction[_P, _R]
    inputs: Mapping[str, FieldSpec]
    outputs: Mapping[str, FieldSpec]
    returns: Returns
    policy: Policy = Policy()
    compute_added_inputs: Callable[..., Mapping[Any, Any]] | None = None
    compute_outputs: Callable[..., Mapping[Any, Any]] | None = None
    from_run: Mapping[str, type] = field(default_factory=dict[str, type])
    hook_parameters: Mapping[HookName, Mapping[str, HookParameter]] = field(default_factory=dict[HookName, Mapping[str, HookParameter]])

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Call the function, as a test or a host would."""
        return self.function(*args, **kwargs)

    @property
    def lineage(self) -> tuple[NodeDefinition, ...]:
        """This version, then each version it replaces, newest first."""
        return (self, *self._replaced)

    def version_of(self, version: int) -> NodeDefinition:
        """This node at ``version``, walking back through what each version replaces; a ``KeyError`` naming the versions
        there are when it has none. ``version in definition.versions`` asks first."""
        return _version_of(self.lineage, version)

    def describe(self) -> NodeDescription:
        """The node as an editor's palette shows it."""
        return self._described(
            is_graph=False, policy=self.policy, inputs=self.inputs.values(), outputs=self.outputs.values(),
            hooks=(self.compute_added_inputs is not None, self.compute_outputs is not None),
        )


@dataclass(frozen=True, eq=False, kw_only=True, repr=False)  # the version's own __repr__
class GraphDefinition(_Version):
    """A version of a node that is a stored graph, run where a graph using it puts its nodes.

    ``@node`` on a record, JSON text or a mapping makes one. It has no function, so calling it is a type error; its
    fields are the graph's, worked out when a graph using it is loaded.

    Attributes:
        graph: The stored graph this node is.
    """

    graph: GraphRecord

    @property
    def lineage(self) -> tuple[NodeDefinition, ...]:
        """This version, then each version it replaces, newest first."""
        return (self, *self._replaced)

    def version_of(self, version: int) -> NodeDefinition:
        """This node at ``version``, walking back through what each version replaces; a ``KeyError`` naming the versions
        there are when it has none. ``version in definition.versions`` asks first."""
        return _version_of(self.lineage, version)

    def describe(self) -> NodeDescription:
        """The node as an editor's palette shows it, with no fields: only ``registry.describe()``, which holds the nodes
        the graph uses, lists them."""
        return self._described(is_graph=True, policy=_DEFAULT_POLICY, inputs=(), outputs=(), hooks=(False, False))


#: One version of a node: a function, or a stored graph.
type NodeDefinition = FunctionDefinition[..., Any] | GraphDefinition


def _version_of(lineage: tuple[NodeDefinition, ...], version: int) -> NodeDefinition:
    for found in lineage:
        if found.version == version:
            return found
    raise KeyError(f"{lineage[0].id!r} has no version {version!r}; it has {', '.join(str(step.version) for step in lineage)}")


class NodeDescription(ConductorModel):
    """A node as an editor's palette shows it: what it is, its versions, its policy and its fields.

    A node that is a stored graph (``is_graph``) offers the fields of the nodes its graph uses, which only a registry
    holds: ``registry.describe()`` lists them, worked out as loading works them out, and the definition's own
    ``describe()`` lists none. ``takes_added_inputs`` and ``computes_outputs`` say its fields depend on what is typed into it.
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
    is_graph: bool
    policy: PolicyDescription
    inputs: list[FieldDescription]
    outputs: list[FieldDescription]
    takes_added_inputs: bool
    computes_outputs: bool


class NodeDecorator(Protocol):
    """What ``@node(...)`` returns: it makes a node of a function, or of a stored graph."""

    @overload
    def __call__(self, target: GraphRecord | str | bytes | Mapping[str, Any], /) -> GraphDefinition: ...

    @overload
    def __call__(self, target: NodeFunction[_P, _R], /) -> FunctionDefinition[_P, _R]: ...

    def __call__(self, target: Any, /) -> NodeDefinition: ...


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
) -> NodeDecorator:
    """Make the function below it, or a stored graph, a node with the id ``id``::

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
      whose stored graphs have their own numbers names them. ``deprecation``
      marks this version going away.
    - ``compute_added_inputs`` and ``compute_outputs`` are hooks run when a
      graph is loaded, which shape the node from what is typed into it.

    A stored graph is a node too: ``node("enrich", title="Enrich", description="...")(record)``, the record given as ``registry.load`` takes one (a
    ``GraphRecord``, JSON text or a mapping), one of the wrong shape an ``InvalidCallError`` of ``invalid_record``.
    """
    if not is_node_id(id):
        raise NodeDeclarationError(
            f"@node takes the node's id first, a non-empty string without '.', '/', '@' or '~', which mean something in an address "
            f"or a waiting id: @node('upper'); got {id!r}",
            code="invalid_node_id",
        )
    if upgrade is not None and replaces is None:
        raise NodeDeclarationError(f"@node({id!r}) has an upgrade but replaces nothing; name the version it upgrades with replaces=", code="upgrade_without_replaces")
    if replaces is not None and replaces.id != id:
        raise NodeDeclarationError(f"@node({id!r}) replaces {replaces!r}; a node replaces an earlier @node of its own id", code="replaces_another_node")
    number = version if version is not None else (replaces.version + 1 if replaces is not None else 1)
    if replaces is not None and number <= replaces.version:
        raise NodeDeclarationError(f"@node({id!r}) is version {number}, not after the version {replaces.version} it replaces; leave version= out, or give one above {replaces.version}", code="version_not_after")

    @overload
    def make(target: GraphRecord | str | bytes | Mapping[str, Any], /) -> GraphDefinition: ...
    @overload
    def make(target: NodeFunction[_P, _R], /) -> FunctionDefinition[_P, _R]: ...
    def make(target: GraphRecord | str | bytes | Mapping[str, Any] | NodeFunction[..., Any], /) -> NodeDefinition:
        if not callable(target):
            given = ", ".join(name for name, value in (("policy", policy), ("compute_added_inputs", compute_added_inputs), ("compute_outputs", compute_outputs)) if value is not None)
            if given:
                raise NodeDeclarationError(
                    f"@node({id!r}) on a stored graph takes no {given}: its nodes run by their own policies and fields; set them there",
                    code="not_for_a_graph",
                )
            record = as_record(target)  # a record, JSON text or a mapping, read as registry.load reads one
            return GraphDefinition(
                graph=record, id=id, version=number, title=title, description=description, category=category,
                tags=tuple(tags), extra=MappingProxyType(dict(extra)), docs=docs, deprecation=deprecation, replaces=replaces,
                upgrade=upgrade or Upgrade(),
            )
        where = f"@node({id!r}) on {target.__name__}: "
        if inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target):
            raise NodeDeclarationError(f"{where}a node is a plain def; the engine gives a slow one threads with Policy(concurrency=)", code="async_run")
        inputs, from_run, outputs, returns = _read_signature(where, target, compute_outputs is not None)
        takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in inspect.signature(target).parameters.values())
        if compute_added_inputs is not None and not takes_kwargs:
            raise NodeDeclarationError(f"{where}a node with compute_added_inputs takes them as **kwargs; add **values to its signature", code="compute_added_inputs_without_var_keyword")
        if compute_added_inputs is None and takes_kwargs:
            raise NodeDeclarationError(f"{where}**kwargs takes the inputs compute_added_inputs adds; the node has no such hook, so remove **kwargs or add compute_added_inputs=", code="var_keyword_without_hook")
        hooks: dict[HookName, Callable[..., Any] | None] = {"compute_added_inputs": compute_added_inputs, "compute_outputs": compute_outputs}
        hook_parameters = {
            kind: _check_hook(where, hook, kind, inputs, adds_inputs=compute_added_inputs is not None)
            for kind, hook in hooks.items() if hook is not None
        }
        definition = FunctionDefinition(
            function=target, inputs=inputs, outputs=outputs, returns=returns, from_run=from_run,
            id=id, version=number, title=title, description=description,
            category=category, tags=tuple(tags), extra=MappingProxyType(dict(extra)), docs=docs, deprecation=deprecation,
            replaces=replaces, upgrade=upgrade or Upgrade(), policy=policy or Policy(),
            compute_added_inputs=compute_added_inputs, compute_outputs=compute_outputs, hook_parameters=hook_parameters,
        )
        if replaces is not None:
            _raise_if_upgrade_is_missing(where, replaces, definition)
        return definition

    return make


#: What a node id never holds: each means something in an address (``node.field``, ``embedding/inner``) or a waiting id
#: (``node@row~fingerprint``).
_MARKS = frozenset("./@~")


def is_node_id(value: str) -> bool:
    """Whether ``value`` can be a node id, of a definition or of a node in a record: a non-empty text without ``. / @ ~``."""
    return bool(value) and not _MARKS & set(value)


def from_run_values(definitions: Iterable[tuple[str, FunctionDefinition[..., Any]]], from_run: Mapping[type, Any]) -> dict[type, Any]:
    """The value each ``FromRun`` parameter of these placed nodes takes, by the class it names; ``InvalidCallError`` listing
    every parameter it can't fill.

    A value keyed by the class itself is taken; else the one value given that is an instance of it, so a subclass
    (a test double, a session of the host's own) is keyed by its own class or its parent's alike.
    """
    given = dict(from_run)
    found: dict[type, Any] = {}
    problems: list[Problem] = []
    for node_id, definition in definitions:
        for name, needed in definition.from_run.items():
            if needed in found:
                continue
            if needed in given:
                found[needed] = given[needed]
                continue
            fits = [value for key, value in given.items() if _fits(needed, key, value)]
            if len(fits) == 1:
                found[needed] = fits[0]
                continue
            names = [key.__name__ for key in given]
            problems.append(make_problem("from_run_ambiguous" if fits else "from_run_missing", node_id, name, needs=needed.__name__, parameter=name, given=names))
    if problems:
        raise InvalidCallError(problems)
    return found


def _fits(needed: type, key: type, value: Any) -> bool:
    """Whether ``value``, given under ``key``, is a ``needed``; a class no ``isinstance`` can ask about (a protocol not
    runtime-checkable) is filled only under its own key."""
    try:
        return isinstance(value, needed) or issubclass(key, needed)
    except TypeError:
        return False


@dataclass(frozen=True, slots=True)
class OneResult:
    """The function returns one value, its output ``result``."""

    def spread(self, returned: Any, outputs: Mapping[str, FieldSpec]) -> tuple[dict[str, Any], bool]:
        """The value as the output ``result``, and whether it needs no checking: one value already of a plain class."""
        spec = outputs["result"]
        return {"result": returned}, spec.shape == "one" and type(returned) in spec.plain_classes


@dataclass(frozen=True, slots=True)
class OutputsClass:
    """The function returns a frozen dataclass, one output per field.

    Attributes:
        cls: The dataclass.
    """

    cls: type

    def spread(self, returned: Any, outputs: Mapping[str, FieldSpec]) -> tuple[dict[str, Any], bool]:
        """Each field of the dataclass returned as its output, to be checked."""
        if not isinstance(returned, self.cls):
            raise FieldValueError(f"the node returns a {self.cls.__name__}, not a {type(returned).__name__}", code="invalid_output")
        return {name: getattr(returned, name) for name in outputs}, False


@dataclass(frozen=True, slots=True)
class HookOutputs:
    """The function returns a mapping of the outputs its ``compute_outputs`` hook gave when the graph loaded."""

    def spread(self, returned: Any, outputs: Mapping[str, FieldSpec]) -> tuple[dict[str, Any], bool]:
        """Each output of the mapping returned, to be checked; one missing or not an output is refused."""
        if not isinstance(returned, Mapping):
            raise FieldValueError(f"the node returns its outputs as a mapping of {', '.join(outputs) or 'nothing'}, not a {type(returned).__name__}", code="invalid_output")
        unknown = sorted(set(returned) - outputs.keys())
        missing = [name for name in outputs if name not in returned]
        if unknown or missing:
            name = (missing or unknown)[0]
            raise FieldValueError(f"'{name}' is {'missing from' if missing else 'not one of'} the outputs it returns", code="invalid_output", name=str(name))
        return {name: returned[name] for name in outputs}, False


@dataclass(frozen=True, slots=True)
class NoOutputs:
    """The function returns ``None``: the node has no outputs."""

    def spread(self, returned: Any, outputs: Mapping[str, FieldSpec]) -> tuple[dict[str, Any], bool]:
        """No outputs; anything but ``None`` returned is refused."""
        if returned is not None:
            raise FieldValueError(f"the node declares no outputs, but returned a {type(returned).__name__}", code="invalid_output")
        return {}, True


#: What a node's function returns, read from its signature at ``@node``.
Returns = OneResult | OutputsClass | HookOutputs | NoOutputs


def _read_signature(
    where: str, function: Callable[..., Any], computed_outputs: bool,
) -> tuple[dict[str, FieldSpec], dict[str, type], dict[str, FieldSpec], Returns]:
    """The inputs, the ``FromRun`` parameters (the class each takes) and the outputs a function's signature declares,
    and what kind of value it returns them as."""
    hints = typing.get_type_hints(function, include_extras=True)
    inputs: dict[str, FieldSpec] = {}
    from_run: dict[str, type] = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if parameter.kind is parameter.VAR_KEYWORD:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.POSITIONAL_ONLY):
            raise NodeDeclarationError(f"{where}{name}: a node's inputs are passed by name; drop the / and *args", code="parameter_not_by_name")
        if name not in hints:
            raise NodeDeclarationError(f"{where}{name}: an input needs a type hint; write one, as it is the input's interface", code="input_without_type")
        default = REQUIRED if parameter.default is inspect.Parameter.empty else parameter.default
        hint = hints[name]
        if marked(hint, _FromRunMarker):
            needed = typing.get_args(hint)[0]
            if not isinstance(needed, type):
                raise NodeDeclarationError(f"{where}{name}: FromRun names the class of the value the run is given, FromRun[Session]", code="from_run_not_a_class")
            from_run[name] = needed
            continue
        inputs[name] = read_field(name, hint, default, where)
        if inputs[name].param.exclusive_group is not None:
            raise NodeDeclarationError(f"{where}{name}: exclusive_group is for outputs; put it on an output's Param", code="exclusive_group_on_input")
    if "return" not in hints:
        raise NodeDeclarationError(f"{where}a node needs a return type; write one, or -> None for no outputs", code="output_without_type")
    returns = hints["return"]
    if computed_outputs:
        return inputs, from_run, {}, HookOutputs()
    if returns is None or returns is type(None):
        return inputs, from_run, {}, NoOutputs()
    if isinstance(returns, type) and dataclasses.is_dataclass(returns):
        if not returns.__dataclass_params__.frozen:  # type: ignore[attr-defined]
            raise NodeDeclarationError(f"{where}a dataclass of outputs is frozen; write @dataclass(frozen=True)", code="outputs_not_frozen")
        own = typing.get_type_hints(returns, include_extras=True)
        outputs = {f.name: read_field(f.name, own[f.name], _declared_default(f), where=where) for f in dataclasses.fields(returns)}
        return inputs, from_run, outputs, OutputsClass(returns)
    return inputs, from_run, {"result": read_field("result", returns, where=where)}, OneResult()


def _declared_default(declared: dataclasses.Field[Any]) -> Any:
    """A dataclass field's default, which a person asked for the output sees proposed; ``REQUIRED`` when it has none."""
    if declared.default is not dataclasses.MISSING:
        return declared.default
    if declared.default_factory is not dataclasses.MISSING:
        return declared.default_factory()
    return REQUIRED


def read_by_hooks(definition: FunctionDefinition[..., Any]) -> frozenset[str]:
    """The inputs a node's hooks read as values: fixed when the graph loads, so never open to a caller."""
    return frozenset(name for parameters in definition.hook_parameters.values() for name, receives in parameters.items() if receives == "value")


def _check_hook(where: str, hook: Callable[..., Any], kind: HookName, inputs: Mapping[str, FieldSpec], *, adds_inputs: bool) -> dict[str, HookParameter]:
    """Reads a hook's parameters: each is passed by name and names an input of the node, but for ``**added: type``.

    Args:
        where: The node and function, to start an error's message with.
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
            raise NodeDeclarationError(f"{where}compute_added_inputs parameter {name!r} reads an arriving type, but inputs are decided before anything arrives; read it in compute_outputs", code="arriving_on_compute_added_inputs")
        if parameter.kind is parameter.VAR_KEYWORD:
            if not (arriving and adds_inputs):
                raise NodeDeclarationError(
                    f"{where}{kind} parameter **{name}: a hook takes ** only as **{name}: type on compute_outputs, for the types arriving on "
                    "the inputs compute_added_inputs added; annotate it type and give the node that hook, or name each parameter after an input",
                    code="hook_parameter_unknown",
                )
            parameters[name] = "added_arriving_types"
        elif parameter.kind is not parameter.POSITIONAL_OR_KEYWORD and parameter.kind is not parameter.KEYWORD_ONLY:
            raise NodeDeclarationError(f"{where}{kind} parameter {name!r}: a hook's parameters are passed by name; name each after an input of the node", code="hook_parameter_unknown")
        elif arriving:
            parameters[name] = "arriving_type"
        elif name not in inputs:
            raise NodeDeclarationError(f"{where}{kind} parameter {name!r} names no input of the node; it has {sorted(inputs)}", code="hook_parameter_unknown")
        else:
            parameters[name] = "value"
    return parameters


def _raise_if_upgrade_is_missing(where: str, older: NodeDefinition, newer: FunctionDefinition[..., Any]) -> None:
    """A field name of the replaced version that is gone from the newer one needs an upgrade to say where it went.

    A gone input is explained by a rename or a ``rewrite``, which rewrites what is typed in; a gone output only by a
    rename, since the edges out of it are what moves.
    """
    match older:
        case GraphDefinition():  # its fields are known only when a graph using it loads
            return
        case FunctionDefinition():
            gone_inputs = set(older.inputs) - set(newer.inputs)
            gone_outputs = set(older.outputs) - set(newer.outputs)
    upgrade = newer.upgrade
    unexplained = gone_outputs - set(upgrade.renamed_outputs)
    if upgrade.rewrite is None:
        unexplained |= gone_inputs - set(upgrade.renamed_inputs)
    if unexplained:
        raise NodeDeclarationError(f"{where}the version it replaces has {sorted(unexplained)}, gone from this one; its Upgrade says where they went", code="upgrade_missing")
