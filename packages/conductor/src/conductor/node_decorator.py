"""``@node`` — a plain function becomes a frozen ``NodeDefinition``.

A node is a function. Its typed signature is its interface::

    @node("upper")
    def upper(text: Text) -> Text:
        \"\"\"Upper-cases a text.\"\"\"
        return Text(text.upper())

``@node`` wraps the function into a value and registers nothing, so
``upper(Text("a"))`` is the unit test and a host decides what to offer by
listing definitions in a ``NodeRegistry``. The title defaults to the
function's name and the description to its docstring.

A node's older versions are listed, oldest first; the decorated function
is the newest, so a version's number is its place and a hole can't be
written. An upgrade sits on the version it leaves::

    def split_v1(text: Text, sep: Text = Text("\\n")) -> Series[Text]: ...

    @node("split", older=[OlderVersion(split_v1, upgrade=Upgrade(renamed_inputs={"sep": "separator"}))])
    def split(text: Text, separator: Text = Text("\\n")) -> Series[Text]: ...

Two hooks shape a node from what it is given in one graph, each a plain
function whose signature says what it reads. A parameter names an input of
the node and receives its typed value, the declared default included; a
parameter marked ``Arriving`` receives the type one call gets there, and
``**x: Arriving`` receives all of them. ``compute_inputs`` returns the
inputs to add, ``compute_outputs`` all the outputs. If a value it needs is
neither typed in nor defaulted, the hook is not called and the node keeps
what it declares.

A host hands over a saved graph as a node with
``NodeDefinition.from_compiled_graphs``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Generic, ParamSpec, TypeVar, get_type_hints

from pydantic import Field
from pydantic.json_schema import SkipJsonSchema

from conductor.dtype import DType, DTypeRef, _AsItArrivesMark, _UntypedValue, is_untyped
from conductor.errors import NodeDeclarationError
from conductor.interface import Interface, open_input_of, read_signature
from conductor.metadata import Input, Output
from conductor.model import ConductorModel

if TYPE_CHECKING:
    from conductor.graph.compiled import CompiledGraph

_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass(frozen=True)
class _ArrivingMark:
    """What makes a hook's parameter a type arriving on an input, not its value."""


#: On a ``compute_outputs`` parameter: the type one call of the node receives
#: on that input, ``Any`` until an edge arrives. ``**x: Arriving`` receives
#: every connected input's.
Arriving = Annotated[type[DType], _ArrivingMark()]


class Deprecation(ConductorModel):
    """A notice that a node, or one of its versions, is going away.

    Every part is optional prose. ``alternative`` is the id of the node to
    use instead (``NodeRegistry`` checks it names something in the same
    registry); ``migration`` explains how to move. It is content an editor
    shows; a graph that uses a deprecated node is not wrong, so this is
    never a ``Problem``.
    """

    header: str | None = None
    description: str | None = None
    alternative: str | None = None
    migration: str | None = None


class Policy(ConductorModel):
    """How the engine runs one version of a node: retries, timeout, concurrency.

    Written by the node author, ``@node("fetch", policy=Policy(retries=3))``,
    and read only by the engine; a person adding the node never sees it.
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


@dataclass(frozen=True)
class Upgrade:
    """How a node saved at one version becomes the next: the fields it renames and a rewrite of its values.

    Sits on the version it leaves. Required only when a field name of that
    version is gone from the next. Read only by ``upgrade_node``; compile
    never upgrades, so a graph pinned at an old version runs that version.

    ``renamed_inputs`` moves each input's binding, lock and author-written
    content to the new name before ``rewrite`` runs. ``rewrite`` takes the
    node's bindings as a dict, a typed-in value as itself and an edge as its
    ``From``, and returns the same shape for the next version.
    ``renamed_outputs`` moves each output's content and rewrites every edge
    in the graph that read the old name.
    """

    renamed_inputs: Mapping[str, str] = field(default_factory=dict)
    renamed_outputs: Mapping[str, str] = field(default_factory=dict)
    rewrite: Callable[[dict[str, Any]], dict[str, Any]] | None = None


@dataclass(frozen=True)
class OlderVersion:
    """One earlier version of a node, as listed in ``@node(older=[...])``."""

    run: Callable[..., Any]
    _: KW_ONLY
    upgrade: Upgrade | None = None
    policy: Policy = Policy()
    deprecation: Deprecation | None = None
    compute_inputs: Callable[..., Iterable[Input]] | None = None
    compute_outputs: Callable[..., Iterable[Output]] | None = None


@dataclass(frozen=True, repr=False)
class _Hook:
    """A ``compute_inputs`` or ``compute_outputs`` function and what its signature reads."""

    function: Callable[..., Iterable[Any]]
    #: The inputs whose typed values it receives; those in ``required`` it can't be called without.
    reads: tuple[str, ...]
    required: frozenset[str]
    #: The inputs whose arriving type it receives by name, and whether it takes all of them.
    arriving: tuple[str, ...]
    all_arriving: bool

    def answer(self, values: Mapping[str, Any], arriving: Mapping[str, Any]) -> tuple[Any, ...] | None:
        """The hook's fields for these values and arriving types; ``None`` when a value it needs has none."""
        if not self.required <= values.keys():
            return None
        arguments = {name: values[name] for name in self.reads if name in values}
        arguments.update({name: arriving.get(name, _UntypedValue) for name in self.arriving})
        if self.all_arriving:
            arguments.update({name: dtype for name, dtype in arriving.items() if name not in arguments})
        return tuple(self.function(**arguments))

    def __repr__(self) -> str:
        return self.function.__qualname__


def _read_hook(function: Callable[..., Iterable[Any]], interface: Interface, kind: str) -> _Hook:
    """``function``'s signature, checked against the version it shapes."""
    hints = get_type_hints(function, include_extras=True)
    names = {inp.name: inp for inp in interface.inputs}
    reads: list[str] = []
    required: set[str] = set()
    arriving: list[str] = []
    all_arriving = False
    for name, parameter in inspect.signature(function).parameters.items():
        hint = hints.get(name)
        marked = getattr(hint, "__metadata__", ()) and any(isinstance(extra, _ArrivingMark) for extra in hint.__metadata__)
        if marked and kind == "compute_inputs":
            raise NodeDeclarationError(f"compute_inputs parameter {name!r} is Arriving, but inputs are decided before anything arrives", code="arriving_on_compute_inputs")
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            if not marked:
                raise NodeDeclarationError(f"{kind} parameter **{name}: the only ** a hook can take is **{name}: Arriving", code="hook_parameter_unknown")
            all_arriving = True
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.POSITIONAL_ONLY):
            raise NodeDeclarationError(f"{kind} parameter {name!r}: a hook's parameters are passed by name", code="hook_parameter_unknown")
        if name not in names:
            raise NodeDeclarationError(f"{kind} parameter {name!r} names no input of the node; it has {sorted(names)}", code="hook_parameter_unknown")
        if marked:
            arriving.append(name)
            continue
        reads.append(name)
        if parameter.default is inspect.Parameter.empty:
            required.add(name)
    return _Hook(function, tuple(reads), frozenset(required), tuple(arriving), all_arriving)


@dataclass(frozen=True, repr=False)
class NodeVersion:
    """One version of a node: its function, ``Interface``, ``Policy``, hooks and notice.

    Kept in ``NodeDefinition.versions`` keyed by number. Read by the
    compiler when a node pins a version and by the engine, which calls
    ``run`` under ``policy``. A version whose body is a graph rather than a
    function is that graph, compiled by the host (a ``CompiledGraph``).
    """

    run: Callable[..., Any]
    interface: Interface
    policy: Policy
    #: This version's own notice; ``None`` means it is not going away.
    deprecation: Deprecation | None = None
    #: How a return splits across the outputs (``ReadSignature.returns``).
    returns: Any = None
    #: For a ``**x: T`` no hook owns: what every edge into the node is an input
    #: like, its name left to the edge. ``None`` otherwise.
    open_input: Input | None = None
    compute_inputs: _Hook | None = None
    compute_outputs: _Hook | None = None
    #: How a node saved at this version moves to the next; ``None`` when nothing moves.
    upgrade_to_next: Upgrade | None = None

    @property
    def hook_reads(self) -> frozenset[str]:
        """The inputs whose typed-in values this version's hooks read."""
        return frozenset(name for hook in (self.compute_inputs, self.compute_outputs) if hook is not None for name in hook.reads)

    def __repr__(self) -> str:
        """The run by its qualified name rather than a function address."""
        notice = "" if self.deprecation is None else f", deprecation={self.deprecation!r}"
        return f"NodeVersion(run={self.run.__qualname__}, interface={self.interface!r}, policy={self.policy!r}{notice})"


def _version(
    run: Callable[..., Any],
    *,
    policy: Policy,
    deprecation: Deprecation | None,
    compute_inputs: Callable[..., Iterable[Input]] | None,
    compute_outputs: Callable[..., Iterable[Output]] | None,
    upgrade: Upgrade | None,
) -> NodeVersion:
    """One ``NodeVersion``, its signature read and its hooks checked against it."""
    if inspect.iscoroutinefunction(run) or inspect.isasyncgenfunction(run):
        raise NodeDeclarationError(f"{run.__name__} is async; a node is a plain function, and the engine gives each call a thread", code="async_run")
    read = read_signature(run)
    outputs_hook = None if compute_outputs is None else _read_hook(compute_outputs, read.interface, "compute_outputs")
    inputs_hook = None if compute_inputs is None else _read_hook(compute_inputs, read.interface, "compute_inputs")
    if outputs_hook is None and read.returns is Mapping:
        raise NodeDeclarationError(f"{run.__name__} returns a Mapping but has no compute_outputs to say which outputs it holds", code="mapping_return_without_hook")
    if outputs_hook is None and any(is_untyped(out.dtype) for out in read.interface.outputs):
        raise NodeDeclarationError(f"{run.__name__} declares an output typed Any but no compute_outputs to type it from what arrives", code="any_output_without_hook")
    open_input = None
    if inputs_hook is not None:
        if read.var_keyword is None:
            raise NodeDeclarationError(f"{run.__name__} has a compute_inputs hook but no **parameter for its inputs to land on", code="compute_inputs_without_var_keyword")
        if any(isinstance(extra, _AsItArrivesMark) for extra in getattr(read.var_keyword[1], "__metadata__", ())):
            raise NodeDeclarationError(f"**{read.var_keyword[0]}: AsItArrives is every edge an input, so a compute_inputs hook has nothing to land on it", code="as_it_arrives_beside_compute_inputs")
    elif read.var_keyword is not None:
        open_input = open_input_of(*read.var_keyword)
    return NodeVersion(
        run=run, interface=read.interface, policy=policy, deprecation=deprecation, returns=read.returns,
        open_input=open_input, compute_inputs=inputs_hook, compute_outputs=outputs_hook, upgrade_to_next=upgrade,
    )


def _check_upgrades(versions: Sequence[NodeVersion]) -> None:
    """A field name of a version that is gone from the next needs an upgrade to say where it went."""
    for older, newer in zip(versions, versions[1:], strict=False):
        gone_inputs = {i.name for i in older.interface.inputs} - {i.name for i in newer.interface.inputs}
        gone_outputs = {o.name for o in older.interface.outputs} - {o.name for o in newer.interface.outputs}
        step = older.upgrade_to_next
        unexplained = gone_inputs - set(step.renamed_inputs) if step is not None and step.rewrite is None else gone_inputs if step is None else set()
        if unexplained or (step is None and gone_outputs):
            raise NodeDeclarationError(
                f"{older.run.__name__} has {sorted(unexplained | gone_outputs)}, gone from the version after it; its Upgrade says where they went",
                code="upgrade_missing",
            )


@dataclass(frozen=True, eq=False, repr=False)
class NodeDefinition(Generic[_P, _R]):
    """A node: what ``@node`` returns, what a registry holds and what a graph names by ``id``.

    Frozen, and callable with the function's own signature, so a node is
    tested by calling it. ``versions`` holds one record per version, keyed
    by number, and ``current`` is the highest.
    """

    #: The registry id. Stored in every graph that uses this node, so
    #: changing it is a data migration, not a rename.
    id: str
    #: What a person sees in the palette. A node copies these when it is
    #: added to a graph and may edit its own copy.
    title: str
    description: str
    #: Where the palette files it. A plain string; the host keeps the table
    #: that titles categories.
    category: str
    tags: tuple[str, ...]
    #: Long-form markdown for the node's help. ``None`` means the
    #: description is all there is to say.
    docs: str | None
    #: The notice that this node is going away; ``None`` means it is not.
    #: A version's own notice sits on its ``NodeVersion``.
    deprecation: Deprecation | None
    versions: Mapping[int, NodeVersion | CompiledGraph]
    current: int

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Call the newest version's function, as a test or a host would."""
        version = self.versions[self.current]
        if not isinstance(version, NodeVersion):
            raise TypeError(f"{self.id!r} is a graph, so it has no function to call; run it through a compiled graph")
        return version.run(*args, **kwargs)

    def __repr__(self) -> str:
        """The record as the call that states what it is, optional parts only when set."""
        parts = [f"id={self.id!r}", f"title={self.title!r}", f"category={self.category!r}"]
        if self.tags:
            parts.append(f"tags={self.tags!r}")
        parts.append(f"versions={tuple(sorted(self.versions))!r}")
        if self.deprecation is not None:
            parts.append(f"deprecation={self.deprecation!r}")
        return f"NodeDefinition({', '.join(parts)})"

    @classmethod
    def from_compiled_graphs(
        cls,
        id: str,
        *,
        title: str,
        description: str,
        category: str = "",
        tags: Sequence[str] = (),
        docs: str | None = None,
        deprecation: Deprecation | None = None,
        versions: Mapping[int, CompiledGraph],
    ) -> NodeDefinition[..., Any]:
        """A node whose versions are graphs a host compiled, by number; its numbers may have holes."""
        if not versions:
            raise NodeDeclarationError(f"{id!r} has no versions; a definition declares at least one", code="versions_empty", node_id=id)
        return NodeDefinition(
            id=id, title=title, description=description, category=category, tags=tuple(tags), docs=docs,
            deprecation=deprecation, versions=MappingProxyType(dict(versions)), current=max(versions),
        )

    def describe(self) -> NodeDescription:
        """This definition as a ``NodeDescription``: what a palette reads."""
        return NodeDescription(
            id=self.id, title=self.title, description=self.description, category=self.category, tags=self.tags,
            docs=self.docs, deprecation=self.deprecation,
            versions={
                number: VersionDescription(
                    inputs=version.interface.inputs,
                    outputs=version.interface.outputs,
                    policy=version.policy if isinstance(version, NodeVersion) else None,
                    open=version.open_input.dtype if isinstance(version, NodeVersion) and version.open_input is not None else None,
                    deprecation=version.deprecation if isinstance(version, NodeVersion) else None,
                )
                for number, version in self.versions.items()
            },
            current=self.current,
        )


def node(
    id: str,
    /,
    *,
    title: str | None = None,
    description: str | None = None,
    category: str = "",
    tags: Sequence[str] = (),
    docs: str | None = None,
    policy: Policy = Policy(),
    deprecation: Deprecation | None = None,
    compute_inputs: Callable[..., Iterable[Input]] | None = None,
    compute_outputs: Callable[..., Iterable[Output]] | None = None,
    older: Sequence[OlderVersion] = (),
) -> Callable[[Callable[_P, _R]], NodeDefinition[_P, _R]]:
    """Declare a node from the function below it (see the module docstring)."""

    def decorate(run: Callable[_P, _R]) -> NodeDefinition[_P, _R]:
        try:
            versions = [
                _version(
                    each.run, policy=each.policy, deprecation=each.deprecation,
                    compute_inputs=each.compute_inputs, compute_outputs=each.compute_outputs, upgrade=each.upgrade,
                )
                for each in older
            ]
            versions.append(_version(run, policy=policy, deprecation=None, compute_inputs=compute_inputs, compute_outputs=compute_outputs, upgrade=None))
            _check_upgrades(versions)
        except NodeDeclarationError as wrong:
            wrong.node_id = id
            raise
        return NodeDefinition(
            id=id,
            title=title or run.__name__.replace("_", " ").strip().capitalize(),
            description=(inspect.getdoc(run) or "").split("\n\n")[0].replace("\n", " ") if description is None else description,
            category=category,
            tags=tuple(tags),
            docs=docs,
            deprecation=deprecation,
            versions=MappingProxyType({number: version for number, version in enumerate(versions, start=1)}),
            current=len(versions),
        )

    return decorate


class VersionDescription(ConductorModel):
    """One version as a palette reads it: its fields, its policy, what an edge
    into it becomes when its inputs are open, and its deprecation notice.

    ``policy`` is ``None`` for a version whose body is a graph, which
    nothing runs as one unit.
    """

    inputs: tuple[Input, ...]
    outputs: tuple[Output, ...]
    policy: Policy | None
    open: DTypeRef | None
    deprecation: Deprecation | None


class NodeDescription(ConductorModel):
    """A node definition as a record — the palette entry.

    Built by ``NodeDefinition.describe()`` on demand and read by an editor:
    a palette is these records dumped through pydantic. Describes the
    *type*. The titles one node in a graph shows live on its ``GraphNode``.
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
