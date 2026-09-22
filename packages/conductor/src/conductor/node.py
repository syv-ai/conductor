"""``NodeDefinition`` — what a node is, and what each of its versions declares.

A node is a class. It declares its identity and what a palette shows
(``id``, ``title``, ``description``, ``category``) and implements ``run``,
whose typed signature is its interface::

    class Upper(NodeDefinition):
        id = "upper"
        title = "Upper case"
        description = "Upper-cases a text."
        category = "text"

        def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())]) -> Annotated[Text, Result(title="Result")]:
            return Text(text.upper())

Several versions live in one class as methods marked ``@version(n)``; the
current one is the method named ``run``. ``@upgrade(1, 2)`` marks the
function that rewrites values saved against version 1 into what version 2
expects; a class with several versions declares one step per adjacent
pair, checked when the class is defined, and ``NodeRegistry.upgraded``
applies them to a graph. ``@deprecated`` marks a class or a version as
going away.

The parts, by when they exist:

* declared when the class is defined — ``NodeVersion`` (a signature, a
  ``Policy`` and the callable), ``GraphVersion`` (a version a host hands
  over by value, whose body is a graph), ``Deprecation``;
* answered per node when a graph is compiled — the two field hooks,
  ``compute_inputs`` and ``compute_outputs``;
* derived on demand for a palette — ``NodeDescription`` and
  ``VersionDescription``, built by ``describe()``.

Nothing here is stored. A registry holds the classes themselves, so a
description is always derived from the live declaration.
"""

from __future__ import annotations

import inspect
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from conductor.interface import Interface
from conductor.metadata import Input, Output
from conductor.model import ConductorModel

if TYPE_CHECKING:
    from conductor.dtype import DType
    from conductor.graph.model import GraphNode



class Deprecation(ConductorModel):
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

class Policy(ConductorModel):
    """How the engine runs one version of a node: retries, timeout, concurrency.

    Written by the node author on the version, ``@version(2, policy=Policy(retries=3))``,
    and read only by the engine; a person placing the node never sees it.
    Worth setting for work that can fail transiently or hang, such as a
    network call. Retrying a pure computation only repeats the same failure,
    which is why only an ``ExternalFailure`` is retried: the node raises one
    itself, or names its client's exception classes in ``retry_on`` and the
    engine wraps those. Every bound is checked when the policy is built.
    """

    #: How many times to re-run after an ``ExternalFailure``. 0 means run once.
    retries: int = Field(default=0, ge=0)

    #: Seconds before the first retry; each later one waits twice as long.
    #: Ignored when ``retries`` is 0.
    delay: float = Field(default=1.0, ge=0)

    #: Seconds the leg waits on one attempt before failing the node with
    #: code ``timeout``. The thread the attempt runs on is not interrupted,
    #: and a timed-out attempt is final. ``None`` means the leg waits.
    timeout: float | None = Field(default=None, gt=0)

    #: How many rows may run at once when the node runs once per row of a
    #: series. ``1`` means one after another. The engine has no other cap.
    concurrency: int = Field(default=8, ge=1)

    #: Exception classes from the node's own client that mean the outside
    #: world failed — a connection error, a timeout, a rate limit. A foreign
    #: exception of one of these classes is wrapped as ``ExternalFailure``
    #: and retried; any other is wrapped as ``NodeExecutionError`` and is
    #: not. Left out of the dump and the schema: a palette reads the policy
    #: as JSON, and a class has no JSON form.
    retry_on: SkipJsonSchema[tuple[type[BaseException], ...]] = Field(default=(), exclude=True)


@dataclass(frozen=True, repr=False)
class NodeVersion:
    """One declared version of a node: its ``run``, ``Interface``, ``Policy`` and notice.

    Built by ``__init_subclass__`` from each method marked ``@version`` (an
    undecorated ``run`` is version 1) and kept in ``NodeDefinition.versions``
    keyed by number, which is why there is no ``number`` field here.
    ``interface`` is derived from ``run``'s signature by ``Interface.of``.
    Read by the registry's numbering check, by the compiler when a
    node pins a version, and by the engine, which calls ``run`` under
    ``policy``. A version whose body is a graph rather than a ``run`` is a
    ``GraphVersion``.
    """

    run: Callable[..., Any]
    interface: Interface
    policy: Policy
    #: This version's own notice, from ``@deprecated`` on its run method;
    #: ``None`` means it is not going away.
    deprecation: Deprecation | None = None

    def __repr__(self) -> str:
        """The run by its qualified name rather than a function address."""
        notice = "" if self.deprecation is None else f", deprecation={self.deprecation!r}"
        return f"NodeVersion(run={self.run.__qualname__}, interface={self.interface!r}, policy={self.policy!r}{notice})"


@dataclass(frozen=True)
class GraphVersion:
    """One version of a definition whose body is a graph rather than a ``run``.

    A host builds one from data — a stored graph it embeds as a node,
    say. ``interface`` is that graph's interface (inputs named by address,
    ``returns`` a ``Mapping``) and ``graph`` the nodes the compiler
    expands under the placing node's name, so the inner nodes run as
    nodes of the outer graph. Nothing runs it as one unit:
    ``NodeRegistry.runner_for`` refuses it and it carries no policy. A
    sibling of ``NodeVersion`` rather than an optional field on it, so
    neither record can be half-filled.
    """

    #: The nodes this version expands to. Edges live in their
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

@dataclass(frozen=True)
class Upgrade:
    """How a node saved at one version becomes the next one: a rewrite of its bindings, and the outputs it renames.

    Declared with ``@upgrade`` in the class body and collected into
    ``cls.upgrades`` when the class is defined, one per adjacent pair of
    versions. Read only by ``NodeRegistry.upgraded``, which runs the chain
    over one node of a graph when a person asks for it; compile never
    upgrades, so a graph pinned at an old version runs that version.

    ``rewrite`` takes the node's bindings as a dict, a typed-in value as
    the value itself and an edge as its ``Edges`` record, and returns the
    same shape for the new version: renaming an input moves its edge with
    it. ``outputs`` maps an old output name to its new one, and the
    registry rewrites every edge in the graph that read the old name.
    """

    rewrite: Callable[[dict[str, Any]], dict[str, Any]]
    outputs: Mapping[str, str]


def upgrade(
    from_version: int, to_version: int, *, outputs: Mapping[str, str] | None = None
) -> Callable[[Callable[..., Any]], staticmethod]:
    """Mark a function as the rewrite from ``from_version`` to ``to_version``.

    It takes the bindings saved against the old version and returns the
    ones the new version expects — a typed-in value as itself, an edge as
    its ``Edges``. ``outputs`` names the outputs the step renames, old to
    new. A ``staticmethod``, because it rewrites data and has no instance
    to consult::

        @upgrade(1, 2)
        def _split_name(values: dict) -> dict:
            first, _, last = values.pop("name").partition(" ")
            return {**values, "first": first, "last": last}

    ``to_version`` is ``from_version + 1``; a longer jump is a chain of steps.
    """

    def decorate(func: Callable[..., Any]) -> staticmethod:
        func.__node_upgrade__ = (from_version, to_version, Upgrade(rewrite=func, outputs=dict(outputs or {})))
        return staticmethod(func)

    return decorate


class VersionDescription(ConductorModel):
    """One version as a palette reads it: its fields, its policy, whether its
    inputs are open (and in which shape) and its deprecation notice.

    One per entry in ``NodeDescription.versions``, built by ``describe()``
    from the ``NodeVersion`` (or ``GraphVersion``) with the callable left
    out. ``policy`` is ``None`` for a graph-bodied version, which nothing
    runs as one unit; ``open`` is the shape of an open interface or ``None``.
    """

    inputs: tuple[Input, ...]
    outputs: tuple[Output, ...]
    policy: Policy | None
    open: Literal["single", "series"] | None
    deprecation: Deprecation | None


class NodeDescription(ConductorModel):
    """A node definition as a record — the palette entry.

    Built by ``NodeDefinition.describe()`` from the class, on demand, and
    read by an editor: a palette is these records dumped through pydantic.
    Never stored and never read back, so there is no copy to keep in step
    with the class.

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

class _NodeMeta(ABCMeta):
    """Gives a node class the repr of what it declares.

    A registry holds classes, so a class is what a person inspects: without
    this it prints as ``<class '__main__.Greet'>``. The rule is pydantic's
    and scikit-learn's — an object prints as the call that states what it
    is, ``Greet(id='greet', title='Greeting', category='text', versions=(1,))``,
    with ``tags`` and ``deprecation`` only when the class has them. A class
    that declares no node (``NodeDefinition`` itself, an intermediate base)
    keeps the class repr.
    """

    def __repr__(cls) -> str:
        if "versions" not in dir(cls):
            return super().__repr__()
        parts = [f"id={cls.id!r}", f"title={cls.title!r}", f"category={cls.category!r}"]
        if cls.tags:
            parts.append(f"tags={tuple(cls.tags)!r}")
        parts.append(f"versions={tuple(sorted(cls.versions))!r}")
        if cls.deprecation is not None:
            parts.append(f"deprecation={cls.deprecation!r}")
        return f"{cls.__name__}({', '.join(parts)})"


class NodeDefinition(ABC, metaclass=_NodeMeta):
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
    #: What a person sees in the palette. A node copies these when it
    #: is added to a graph and may edit its own copy.
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
    #: Every step the class declares with ``@upgrade``, keyed by the pair it
    #: spans, ``(from_version, to_version)``. Collected when the class is
    #: defined, from the class and its bases, and checked against the
    #: versions: one step per adjacent pair, each forward by one.
    #: ``NodeRegistry.upgraded`` runs them. Derived only: unlike
    #: ``versions`` it is never given, so a class body that sets it is
    #: refused, and an ``@upgrade`` attached after definition is not seen.
    upgrades: ClassVar[dict[tuple[int, int], Upgrade]]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Check the declaration, derive its versions and collect its upgrades when the class is defined.

        Subclassing is the trigger, so it cannot be forgotten and a
        malformed node fails at import with the traceback at the class. A
        class that sets ``versions`` in its own body is taken as given —
        its versions came by value and there is no ``run`` to walk.
        Numbering rules (from 1, no holes) are the registry's, not the
        class's, and live in ``register()``.
        """
        super().__init_subclass__(**kwargs)
        if "upgrades" in vars(cls):
            raise TypeError(
                f"{cls.__name__} sets 'upgrades'; it is collected from the class's "
                "@upgrade methods, so declare each rewrite with @upgrade"
            )
        cls.upgrades = cls._collect_upgrades()
        given = "versions" in vars(cls)
        if not given and "id" in vars(cls) and "run" not in vars(cls):
            defined = sorted(n for n, v in vars(cls).items() if callable(v) and not n.startswith("__"))
            raise TypeError(
                f"{cls.__name__} declares an id but no run; a node's work is the method named run "
                f"(the class defines {defined})"
            )
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
        cls._derive_versions()
        cls._check_upgrades()
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

    @classmethod
    def _derive_versions(cls) -> None:
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
        elif getattr(cls.run, "__node_version__", None) is None:
            # Beside declared versions a plain ``run`` has no number, and
            # the current version is the method named ``run``.
            raise TypeError(
                f"{cls.__name__}: run has no @version, but other methods do; "
                "mark run with the number of the version it is"
            )

        for number, fn in methods.items():
            if inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn):
                # The engine calls ``run`` in a worker thread and never awaits
                # it, so an async one would hand back a coroutine as its result.
                raise TypeError(
                    f"{cls.__name__}: version {number} is async; run is a plain "
                    "function, and the engine gives each call a thread"
                )

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

    @classmethod
    def _collect_upgrades(cls) -> dict[tuple[int, int], Upgrade]:
        """Every step this class and its bases declare, keyed by the pair it
        spans — what ``__init_subclass__`` stores on ``cls.upgrades``. Two
        steps for one pair in one class body are refused; a subclass's step
        replaces its base's."""
        found: dict[tuple[int, int], Upgrade] = {}
        for klass in reversed(cls.__mro__):
            claimed: set[tuple[int, int]] = set()
            for name in vars(klass):
                # `getattr`, not `vars()[name]`: a `staticmethod` descriptor does
                # not forward attribute lookups to the function it wraps, so the
                # marker is invisible from the outside.
                marker = getattr(getattr(cls, name, None), "__node_upgrade__", None)
                if marker is None:
                    continue
                from_version, to_version, step = marker
                if (from_version, to_version) in claimed:
                    raise TypeError(
                        f"{cls.__name__}: two methods declare @upgrade({from_version}, {to_version}); "
                        "one step per pair of versions"
                    )
                claimed.add((from_version, to_version))
                found[(from_version, to_version)] = step
        return found

    @classmethod
    def _check_upgrades(cls) -> None:
        """Refuse a step that is not one version forward between two declared
        versions, and an adjacent pair of versions with no step, naming it."""
        for from_version, to_version in cls.upgrades:
            if to_version != from_version + 1 or not {from_version, to_version} <= cls.versions.keys():
                raise TypeError(
                    f"{cls.__name__}: @upgrade({from_version}, {to_version}) is not a step between "
                    f"two adjacent versions it declares ({sorted(cls.versions)}); each step goes "
                    "one version forward, and a longer jump is a chain of steps"
                )
        for number in sorted(cls.versions):
            if number + 1 in cls.versions and (number, number + 1) not in cls.upgrades:
                raise TypeError(
                    f"{cls.__name__} declares versions {number} and {number + 1} but no "
                    f"@upgrade({number}, {number + 1}); a graph saved at {number} could never move on"
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

        Override when the node's fields depend on configuration — a mode
        dropdown that changes which fields exist. The default returns
        ``declared``. ``declared`` is passed in rather than read off the
        class because the placement pins a version, which may not be the
        newest; ``values`` are the values the author typed (a connected input
        has no value until the graph runs).
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
        connected input. ``arriving`` maps each connected input name to the type
        one call receives there — for a series into a scalar input, its
        element type — and has no entry for an unconnected input. It is a
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
