"""``NodeDefinition`` — what a node is, and what each of its versions declares.

A node is a class. It declares its identity and what a palette shows
(``id``, ``title``, ``description``, ``category``) and implements ``run``,
whose typed signature is its interface::

    class Upper(NodeDefinition):
        id = "upper"
        title = "Upper case"
        description = "Upper-cases a text."
        category = "text"

        def run(self, text: Annotated[Text, Textarea(title="Text")]) -> Annotated[Text, Result(title="Result")]:
            return Text(text.upper())

Several versions live in one class as methods marked ``@version(n)``; the
current one is the method named ``run``. ``@upgrade(1, 2)`` marks the
function that rewrites values saved against version 1 into what version 2
expects. ``@deprecated`` marks a class or a version as going away.

The parts, by when they exist:

* declared when the class is defined — ``NodeVersion`` (a signature, a
  ``Policy`` and the callable), ``GraphVersion`` (a version a host hands
  over by value, whose body is a graph), ``Deprecation``;
* answered per placement when a flow is compiled — the two roster hooks,
  ``compute_inputs`` and ``compute_outputs``;
* derived on demand for a palette — ``NodeDescription`` and
  ``VersionDescription``, built by ``describe()``.

Nothing here is stored. A registry holds the classes themselves, so a
description is always derived from the live declaration.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal

from conductor.interface import Interface
from conductor.metadata import Input, Output
from conductor.types import NodeCategory

if TYPE_CHECKING:
    from conductor.dtype import DType
    from conductor.graph.model import GraphNode


class Refuses(Exception):
    """Raised by a roster hook that cannot answer for the values it was given.

    ``code`` and ``message`` are the host's, and the compiler reports them
    as the placement's problem — the same shape as ``DType.refuses_whole``.
    """

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)

@dataclass(frozen=True, kw_only=True)
class Deprecation:
    """A notice that a node, or one of its versions, is going away.

    Every part is optional prose. ``alternative`` is the id of the node to
    use instead (``register()`` checks it names something in the same
    registry); ``migration`` explains how to move.

    Set by ``@deprecated`` on a class (the whole node) or on a ``@version``
    run method (that version only). It is content an editor shows; a graph
    that places a deprecated node is not wrong, so this is never a
    ``Problem``. The record's presence is the whole fact — there is no
    ``deprecated: bool`` beside it.
    """

    header: str | None = None
    description: str | None = None
    alternative: str | None = None
    migration: str | None = None


def deprecated(
    *,
    header: str | None = None,
    description: str | None = None,
    alternative: str | None = None,
    migration: str | None = None,
) -> Callable[[Any], Any]:
    """Mark a node class, or one ``@version`` run method, as going away.

    On the class the whole node retires; on a run method only that version
    does. Both notices are shown when both exist. A decorator rather than
    a class attribute because absence means "not deprecated", which is
    the right default.
    """
    notice = Deprecation(
        header=header, description=description, alternative=alternative, migration=migration
    )

    def decorate(target: Any) -> Any:
        if isinstance(target, type):
            if not issubclass(target, NodeDefinition):
                raise TypeError(f"@deprecated marks a NodeDefinition or a run method, not {target!r}")
            target.deprecation = notice
        else:
            target.__node_deprecation__ = notice
        return target

    return decorate

@dataclass(frozen=True)
class Policy:
    """How the engine runs one version of a node: retries, timeout, concurrency.

    Written by the node author on the version, ``@version(2, policy=Policy(retries=3))``,
    and read only by the engine; a person placing the node never sees it.
    Worth setting for work that can fail transiently or hang, such as a
    network call. Retrying a pure computation only repeats the same failure.
    """

    #: How many times to re-run after a failure. 0 means run once.
    retries: int = 0

    #: Seconds between attempts. Ignored when ``retries`` is 0.
    delay: float = 1.0

    #: Seconds after which the engine abandons the node and fails it.
    #: ``None`` means the run's own timeout is the only limit.
    timeout: float | None = None

    #: How many rows may run at once when the node runs once per row of a
    #: series. ``1`` means one after another. The engine has no other cap.
    concurrency: int = 8


@dataclass(frozen=True)
class NodeVersion:
    """One declared version of a node: its ``run``, ``Interface``, ``Policy`` and notice.

    Built by ``__init_subclass__`` from each method marked ``@version`` (an
    undecorated ``run`` is version 1) and kept in ``NodeDefinition.versions``
    keyed by number, which is why there is no ``number`` field here.
    ``interface`` is derived from ``run``'s signature by ``Interface.of``.
    Read by the registry's numbering check, by the compiler when a
    placement pins a version, and by the engine, which calls ``run`` under
    ``policy``. A version whose body is a graph rather than a ``run`` is a
    ``GraphVersion``.
    """

    run: Callable[..., Any]
    interface: Interface
    policy: Policy
    #: This version's own notice, from ``@deprecated`` on its run method;
    #: ``None`` means it is not going away.
    deprecation: Deprecation | None = None


@dataclass(frozen=True)
class GraphVersion:
    """One version of a definition whose body is a graph rather than a ``run``.

    A host builds one from data — an embedded flow's approved version,
    say. ``interface`` is that flow's interface (inputs named by address,
    ``returns`` a ``Mapping``) and ``graph`` the placements the compiler
    expands under the placing node's name, so the inner nodes run as
    nodes of the outer flow. Nothing runs it as one unit: ``runner_for``
    refuses it and it carries no policy. A sibling of ``NodeVersion``
    rather than an optional field on it, so neither record can be
    half-filled.
    """

    #: The placements this version expands to. Wiring lives in their
    #: bindings, so the nodes are the whole graph.
    graph: tuple[GraphNode, ...]
    interface: Interface


def version(
    number: int, *, policy: Policy | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a method as one version of this node's ``run``.

    The current version is the method named ``run``; older ones keep any
    name that reads well, conventionally ``run_v1``. The signature and the
    policy both belong to the version. A node with a single version needs
    no decorator.

    This only marks the method; ``__init_subclass__`` builds a
    ``NodeVersion`` per marked method.
    """

    def decorate(method: Callable[..., Any]) -> Callable[..., Any]:
        method.__node_version__ = number
        method.__node_policy__ = policy or Policy()
        return method

    return decorate

def upgrade(
    from_version: int, to_version: int
) -> Callable[[Callable[..., Any]], staticmethod]:
    """Mark a function as the value rewrite from ``from_version`` to ``to_version``.

    It takes the values saved against the old version and returns the
    values the new one expects. A ``staticmethod``, because it rewrites
    data and has no instance to consult::

        @upgrade(1, 2)
        def _split_name(values: dict) -> dict:
            first, _, last = values["name"].partition(" ")
            return {**values, "first": first, "last": last}
    """

    def decorate(func: Callable[..., Any]) -> staticmethod:
        func.__node_upgrade__ = (from_version, to_version)
        return staticmethod(func)

    return decorate


def upgrade_methods(cls: type) -> dict[tuple[int, int], Callable[..., Any]]:
    """Every value rewrite this class declares, keyed by the step it spans."""
    found: dict[tuple[int, int], Callable[..., Any]] = {}
    for klass in reversed(cls.__mro__):
        for name in vars(klass):
            # `getattr`, not `vars()[name]`: a `staticmethod` descriptor does
            # not forward attribute lookups to the function it wraps, so the
            # marker is invisible from the outside.
            fn = getattr(cls, name, None)
            step = getattr(fn, "__node_upgrade__", None)
            if step is not None:
                found[step] = fn
    return found

@dataclass(frozen=True, kw_only=True)
class VersionDescription:
    """One version as a palette reads it: fields, policy, roster shape and notice.

    One per entry in ``NodeDescription.versions``, built by ``describe()``
    from the ``NodeVersion`` (or ``GraphVersion``) with the callable left
    out. ``policy`` is ``None`` for a graph-bodied version, which nothing
    runs as one unit; ``open`` is the open roster's shape or ``None``.
    """

    inputs: tuple[Input, ...]
    outputs: tuple[Output, ...]
    policy: Policy | None
    open: Literal["single", "series"] | None
    deprecation: Deprecation | None


@dataclass(frozen=True, kw_only=True)
class NodeDescription:
    """A node definition as a record — the palette entry.

    Built by ``NodeDefinition.describe()`` from the class, on demand, and
    read by an editor: a palette is these records dumped through pydantic.
    Never stored, so there is no copy to keep in step with the class.

    Describes the *type*. The titles a particular placement shows live on
    its ``GraphNode``.
    """

    id: str
    title: str
    description: str
    category: str
    tags: tuple[str, ...]
    docs: str | None
    deprecation: Deprecation | None
    versions: dict[int, VersionDescription]
    current: int

class NodeDefinition(ABC):
    """Base class for every node.

    A subclass declares ``id``, ``title``, ``description`` and ``category``
    and implements ``run``; the class is checked and its versions derived
    the moment it is defined. Nothing here tells the engine what to do
    with the node: a node that skips an output returns ``SKIPPED`` on it, a
    node that needs a person's answer returns ``Asks``, and the engine
    acts on the value.
    """

    # --- identity -------------------------------------------------------
    #: The registry id. Stored in every graph that places this node, so
    #: changing it is a data migration, not a rename.
    id: ClassVar[str]

    # --- what a person reads ---------------------------------------------
    #: What a person sees in the palette. A placement copies these when it
    #: is added to a flow and may edit its own copy.
    title: ClassVar[str]
    description: ClassVar[str]
    #: Where the palette files it. A plain string; the host keeps the table
    #: that titles categories. Required, so a node cannot land in a
    #: default section by accident.
    category: ClassVar[str]
    tags: ClassVar[tuple[str, ...]] = ()

    #: Long-form markdown for the node's help. ``None`` means the
    #: description is all there is to say.
    docs: ClassVar[str | None] = None

    #: The notice that this node is going away, set by ``@deprecated`` on
    #: the class; ``None`` means it is not. A version's own notice sits on
    #: its ``NodeVersion``.
    deprecation: ClassVar[Deprecation | None] = None

    # --- derived, or given --------------------------------------------------
    #: One record per declared version, keyed by number, and the highest
    #: number. Derived when the class is defined from each marked ``run``
    #: method — unless the class sets ``versions`` itself, as a host does
    #: when it builds a definition from data and hands over
    #: ``GraphVersion`` records by value.
    versions: ClassVar[dict[int, "NodeVersion | GraphVersion"]]
    current: ClassVar[int]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check the declaration and derive its versions when the class is defined.

        Subclassing is the trigger, so it cannot be forgotten and a
        malformed node fails at import with the traceback at the class. A
        class that sets ``versions`` in its own body is taken as given —
        its versions came by value and there is no ``run`` to walk.
        Numbering rules (from 1, no holes) are the registry's, not the
        class's, and live in ``register()``.
        """
        super().__init_subclass__(**kwargs)
        given = "versions" in vars(cls)
        if not given and (inspect.isabstract(cls) or "run" not in vars(cls)):
            # An intermediate base that adds no ``run`` — and hands over no
            # ``versions`` — declares nothing.
            return
        for required in ("id", "title", "description", "category"):
            if not getattr(cls, required, None):
                raise TypeError(
                    f"{cls.__name__} must declare a class-level '{required}'"
                )
        if given:
            # Given by value: nothing to walk, and no ``run`` needed, since
            # a graph-bodied version is expanded by the compiler.
            if not cls.versions:
                raise TypeError(
                    f"{cls.__name__}.versions is empty; a definition declares at least one version"
                )
            cls.current = max(cls.versions)
            return
        _derive_versions(cls)
        if cls.compute_outputs is NodeDefinition.compute_outputs and any(
            out.dtype is Any
            for version in cls.versions.values()
            if isinstance(version, NodeVersion)
            for out in version.interface.outputs
        ):
            # An ``Any`` output promises that ``compute_outputs`` will type
            # it from what arrives; without the hook, refuse here rather
            # than three nodes downstream.
            raise TypeError(
                f"{cls.__name__} declares an output typed Any but no "
                "compute_outputs to type it from what arrives"
            )

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Any:
        """The node's work. Its typed signature is the declared interface."""

    # --- optional shaping hooks -----------------------------------------
    # The only home for node-specific knowledge: generic code never names
    # a node type; a node that needs a rule declares it here, on itself.

    def compute_inputs(
        self, declared: tuple[Input, ...], values: Mapping[str, Any]
    ) -> tuple[Input, ...]:
        """The inputs one placement of this node actually has.

        Override when the roster depends on configuration — a mode
        dropdown that changes which fields exist. The default returns
        ``declared``. ``declared`` is passed in rather than read off the
        class because the placement pins a version, which may not be the
        newest; ``values`` are the values the author typed (a wired input
        has no value until the flow runs).
        """
        return declared

    def compute_outputs(
        self,
        declared: tuple[Output, ...],
        values: Mapping[str, Any],
        arriving: Mapping[str, type[DType]],
    ) -> tuple[Output, ...]:
        """The outputs one placement of this node actually has.

        Override when the outputs come from a value (a sheet's header
        row, a schema the author built) or from the *type* arriving on a
        wired input. ``arriving`` maps each wired input name to the type
        one call receives there — for a series into a scalar input, its
        element type — and has no entry for an unwired input. It is a
        type, never a value. The default returns ``declared``.
        """
        return declared

    @classmethod
    def describe(cls) -> NodeDescription:
        """This definition as a ``NodeDescription`` — the one serialisation of a node.

        A classmethod, because a description is of the type; what a
        placement adds (its titles, its bindings) lives on the ``GraphNode``.
        """
        return NodeDescription(
            id=cls.id,
            title=cls.title,
            description=cls.description,
            category=cls.category,
            tags=cls.tags,
            docs=cls.docs,
            deprecation=cls.deprecation,
            versions={
                number: VersionDescription(
                    inputs=v.interface.inputs,
                    outputs=v.interface.outputs,
                    policy=v.policy if isinstance(v, NodeVersion) else None,
                    open=v.interface.open,
                    deprecation=v.deprecation if isinstance(v, NodeVersion) else None,
                )
                for number, v in cls.versions.items()
            },
            current=cls.current,
        )

def _derive_versions(cls: type) -> None:
    """Collect every version this class declares into ``cls.versions``.

    Reads ``vars()`` of the class and its bases rather than ``dir()``, so an
    inherited version is found once and an override replaces it.
    """
    methods: dict[int, Callable[..., Any]] = {}
    policies: dict[int, Policy] = {}

    for klass in reversed(cls.__mro__):
        # One class body at a time, so "two methods claim this version" is
        # judged *within* a class. Across the MRO a later entry is a
        # subclass overriding a version its base declared.
        claimed: dict[int, Callable[..., Any]] = {}
        for attr in vars(klass).values():
            fn = attr.__func__ if isinstance(attr, staticmethod) else attr
            number = getattr(fn, "__node_version__", None)
            if number is None:
                continue
            if number in claimed:
                raise TypeError(
                    f"{cls.__name__}: two methods claim version {number}. "
                    "Each version is one signature."
                )
            claimed[number] = fn
        for number, fn in claimed.items():
            methods[number] = fn
            policies[number] = fn.__node_policy__

    if not methods:
        # An undecorated `run` is version 1 with the default policy.
        methods[1] = cls.run
        policies[1] = Policy()

    # No contiguity check here: numbering from 1 with no holes is the
    # registry's rule and lives in ``register()``. A definition a host
    # loads from data may legitimately carry only the versions {1, 3}.

    cls.versions = {
        n: NodeVersion(
            run=fn,
            interface=Interface.of(fn),
            policy=policies[n],
            # ``@deprecated`` on a run method stamps the marker.
            deprecation=getattr(fn, "__node_deprecation__", None),
        )
        for n, fn in methods.items()
    }
    cls.current = max(methods)


class BaseNode(ABC):
    """The old class-based contract. Deleted with the decorator at the end of this plan."""

    node_id: ClassVar[str]
    node_name: ClassVar[str]
    node_description: ClassVar[str]
    node_version: ClassVar[int] = 1
    node_tags: ClassVar[tuple[str, ...]] = ()
    node_category: ClassVar[NodeCategory] = NodeCategory.IO

    @abstractmethod
    def execute(self, req: Any) -> Any:
        """Execute the node. Receives a NodeExecRequest."""
