"""``@node`` — a plain function becomes a frozen ``NodeDefinition``.

A node is a function. Its typed signature is its interface::

    @node("upper")
    def upper(text: Text) -> Text:
        \"\"\"Upper-cases a text.\"\"\"
        return Text(text.upper())

``@node`` wraps the function into a value and registers nothing, so
``upper(Text("a"))`` is the unit test and a host decides what to offer by
adding definitions to a ``NodeRegistry`` (whose ``@registry.node`` takes
the same arguments and adds the node too). The title defaults to the
function's name and the description to its docstring. ``extra`` is the
host's own JSON about the node, which conductor keeps and never reads.

A newer version of a node names the one it replaces, itself a ``@node`` of the
same id, and says how a node saved at that one moves here. A version's
number is its place in that chain, so a hole can't be written::

    @node("split")
    def split_v1(text: Text, sep: Text = Text("\\n")) -> Series[Text]: ...

    @node("split", replaces=split_v1, upgrade=Upgrade(renamed_inputs={"sep": "separator"}))
    def split(text: Text, separator: Text = Text("\\n")) -> Series[Text]: ...

Only the newest goes in a registry, and only its title, description and
category are read. A setting of one version (its ``policy``, ``deprecation``
and hooks) is written on that version's ``@node``.

Two hooks shape a node from what it is given in one graph, each a plain
function whose signature says what it reads. A parameter names an input of
the node and receives its value, the static or the declared default; a
parameter marked ``ArrivingType`` receives the type one call gets there, the
edge's or the static's, and ``**x: ArrivingType`` receives all of them. ``compute_added_inputs`` returns the
inputs to add, ``compute_outputs`` all the outputs. If a value it needs is
neither set nor defaulted, the hook is not called and the node keeps
what it declares.

A host hands over a saved graph as a node with
``NodeDefinition.from_compiled_graphs``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Generic, ParamSpec, TypeVar, get_type_hints

from pydantic import Field, JsonValue, TypeAdapter, ValidationError
from pydantic.json_schema import SkipJsonSchema

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.dtype import (
    DType,
    DTypeField,
    _AsItArrivesMark,
    _UntypedValue,
    dtype_of,
    is_untyped,
)
from conductor._internal.errors import NodeDeclarationError
from conductor._internal.field_records import Input, Output, humanised
from conductor._internal.graph.binding import OPEN, From, Static
from conductor._internal.graph.graph_records import GraphNode
from conductor._internal.node_signature import (
    Interface,
    input_template_for_var_keyword,
    read_signature,
)
from conductor._internal.table import is_frame, type_of_frame

if TYPE_CHECKING:
    from conductor._internal.graph.compiled_graph import CompiledGraph

_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass(frozen=True)
class _ArrivingMark:
    """What makes a hook's parameter a type arriving on an input, not its value."""


#: On a ``compute_outputs`` parameter: the type one call of the node receives
#: on that input: the edge's, or the static's when no edge is there,
#: ``Any`` until something is. ``**x: ArrivingType`` receives every connected input's.
ArrivingType = Annotated[type[DType], _ArrivingMark()]


class Deprecation(ConductorModel):
    """A notice that a version of a node is going away; on the newest version, that the node is.

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
    retry_delay: float = Field(default=1.0, ge=0)

    #: Seconds the leg waits on one attempt before failing the node with
    #: code ``timeout``. The thread the attempt runs on is not interrupted,
    #: and a timed-out attempt is final. ``None`` means the leg waits.
    attempt_timeout: float | None = Field(default=None, gt=0)

    #: How many rows may run at once when the node runs once per row of a
    #: series. ``1`` means one after another. The engine has no other cap.
    concurrency: int = Field(default=8, ge=1)

    #: ``False`` for a node whose result depends on the world, not on its
    #: inputs (a file read by its path, today's exchange rate): a new leg from a
    #: ``complete`` state calls it again, and an earlier unit that received the
    #: same inputs never stands in for it. A leg going on from a waiting, failed
    #: or cancelled state keeps what it read.
    reuse_when_inputs_match: bool = True

    #: Exception classes from the node's own client that mean the outside
    #: world failed — a connection error, a timeout, a rate limit. A foreign
    #: exception of one of these classes is wrapped as ``ExternalFailure``
    #: and retried; any other is the code ``execution_failed`` and is
    #: not. Left out of the dump and the schema: an editor reads the policy
    #: as JSON, and a class has no JSON form.
    retry_on: SkipJsonSchema[tuple[type[BaseException], ...]] = Field(default=(), exclude=True)


@dataclass(frozen=True)
class Upgrade:
    """How a node saved at one version becomes the next: the fields it renames and a rewrite of its values.

    Given on the newer version's ``@node``, beside ``replaces``. Required only
    when a field name of the replaced version is gone from the newer one. Read only by ``upgrade_node``; compile
    never upgrades, so a graph pinned at an old version runs that version.

    ``renamed_inputs`` moves each input's binding, lock and author-written
    content to the new name before ``rewrite`` runs. ``rewrite`` takes the
    node's bindings as a dict, a static as itself and an edge as its
    ``From``, and returns the same shape for the next version.
    ``renamed_outputs`` moves each output's content and rewrites every edge
    in the graph that read the old name.
    """

    renamed_inputs: Mapping[str, str] = field(default_factory=dict)
    renamed_outputs: Mapping[str, str] = field(default_factory=dict)
    rewrite: Callable[[dict[str, Any]], dict[str, Any]] | None = None


@dataclass(frozen=True, repr=False)
class _ComputeFieldsHook:
    """A ``compute_added_inputs`` or ``compute_outputs`` function and what its signature reads."""

    function: Callable[..., Iterable[Any]]
    #: The inputs whose values it receives; those in ``inputs_it_cannot_run_without`` it can't be called without.
    inputs_it_reads: tuple[str, ...]
    inputs_it_cannot_run_without: frozenset[str]
    #: The inputs whose arriving type it receives by name, and whether it takes all of them.
    inputs_whose_arriving_type_it_reads: tuple[str, ...]
    reads_every_arriving_type: bool
    #: What each input it reads holds when nothing sets it, for the inputs that have a default.
    defaults: Mapping[str, Any]

    def call_with(self, statics: Mapping[str, Any], arriving_types: Mapping[str, Any]) -> tuple[Any, ...] | None:
        """The hook's fields for these statics and arriving types; ``None`` when a value it needs has none.

        An ``ArrivingType`` parameter gets the edge's type, or the type of the
        static when no edge is there; ``Any`` until something is."""
        values = {**self.defaults, **statics}
        if not self.inputs_it_cannot_run_without <= values.keys():
            return None
        arguments = {name: values[name] for name in self.inputs_it_reads if name in values}
        for name in self.inputs_whose_arriving_type_it_reads:
            static_value: Any = statics.get(name)
            arguments[name] = arriving_types.get(name, type_of_frame(static_value) if is_frame(static_value) else (dtype_of(type(static_value)) or _UntypedValue))
        if self.reads_every_arriving_type:
            arguments.update({name: dtype for name, dtype in arriving_types.items() if name not in arguments})
        return tuple(self.function(**arguments))

    def __repr__(self) -> str:
        return self.function.__qualname__


def _parse_hook(function: Callable[..., Iterable[Any]], interface: Interface, kind: str) -> _ComputeFieldsHook:
    """``function``'s signature, checked against the version it shapes."""
    hints = get_type_hints(function, include_extras=True)
    names = {input_.name: input_ for input_ in interface.inputs}
    reads: list[str] = []
    required: set[str] = set()
    arriving: list[str] = []
    reads_every_arriving_type = False
    for name, parameter in inspect.signature(function).parameters.items():
        hint = hints.get(name)
        is_arriving = getattr(hint, "__metadata__", ()) and any(isinstance(extra, _ArrivingMark) for extra in hint.__metadata__)
        if is_arriving and kind == "compute_added_inputs":
            raise NodeDeclarationError(f"compute_added_inputs parameter {name!r} is ArrivingType, but inputs are decided before anything arrives", code="arriving_on_compute_added_inputs")
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            if not is_arriving:
                raise NodeDeclarationError(f"{kind} parameter **{name}: the only ** a hook can take is **{name}: ArrivingType", code="hook_parameter_unknown")
            reads_every_arriving_type = True
            continue
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.POSITIONAL_ONLY):
            raise NodeDeclarationError(f"{kind} parameter {name!r}: a hook's parameters are passed by name", code="hook_parameter_unknown")
        if name not in names:
            raise NodeDeclarationError(f"{kind} parameter {name!r} names no input of the node; it has {sorted(names)}", code="hook_parameter_unknown")
        if is_arriving:
            arriving.append(name)
            continue
        reads.append(name)
        if parameter.default is inspect.Parameter.empty:
            required.add(name)
    return _ComputeFieldsHook(function, tuple(reads), frozenset(required), tuple(arriving), reads_every_arriving_type, {name: names[name].default for name in reads if not names[name].required})


@dataclass(frozen=True, repr=False)
class NodeVersion:
    """One version of a node: its body, ``Interface``, ``Policy``, hooks and notice.

    Kept in ``NodeDefinition.versions`` keyed by number. Read by the
    compiler when a node pins a version and by the engine, which calls
    ``body`` under ``policy``. The body is a function, or a graph a host
    compiled, whose interface is what it offers.
    """

    body: Callable[..., Any] | CompiledGraph
    interface: Interface
    #: How the engine runs a function; ``None`` on a graph, which nothing runs as one unit.
    policy: Policy | None
    #: The notice that this version is going away; ``None`` means it is not.
    deprecation: Deprecation | None = None
    #: The ``FromRun`` parameters: the value of each type goes under this name in a call.
    from_run: Mapping[str, type] = field(default_factory=dict)
    #: How a return splits across the outputs (``ParsedSignature.return_shape``).
    return_shape: Any = None
    #: For a ``**x: T`` no hook owns: what every edge into the node is an input
    #: like, its name left to the edge. ``None`` otherwise.
    open_input_template: Input | None = None
    compute_added_inputs: _ComputeFieldsHook | None = None
    compute_outputs: _ComputeFieldsHook | None = None
    #: How a node saved at the version before this one moves here; ``None`` when nothing moves.
    upgrade_from_previous: Upgrade | None = None

    @property
    def embedded_graph(self) -> CompiledGraph | None:
        """The compiled graph this version is; ``None`` when its body is a function."""
        return None if callable(self.body) else self.body

    def __repr__(self) -> str:
        """A function body by its qualified name rather than an address."""
        notice = "" if self.deprecation is None else f", deprecation={self.deprecation!r}"
        body = self.body.__qualname__ if callable(self.body) else repr(self.body)
        return f"NodeVersion(body={body}, interface={self.interface!r}, policy={self.policy!r}{notice})"


def _version_of_function(
    function: Callable[..., Any], policy: Policy, deprecation: Deprecation | None,
    compute_added_inputs: Callable[..., Iterable[Input]] | None, compute_outputs: Callable[..., Iterable[Output]] | None, upgrade: Upgrade | None,
) -> NodeVersion:
    """One ``NodeVersion`` of a function, its signature read and its hooks checked against it."""
    if inspect.iscoroutinefunction(function) or inspect.isasyncgenfunction(function):
        raise NodeDeclarationError(f"{function.__name__} is async; a node is a plain function, and the engine gives each call a thread", code="async_run")
    signature = read_signature(function)
    outputs_hook = None if compute_outputs is None else _parse_hook(compute_outputs, signature.interface, "compute_outputs")
    inputs_hook = None if compute_added_inputs is None else _parse_hook(compute_added_inputs, signature.interface, "compute_added_inputs")
    if outputs_hook is None and signature.return_shape is Mapping:
        raise NodeDeclarationError(f"{function.__name__} returns a Mapping but has no compute_outputs to say which outputs it holds", code="mapping_return_without_hook")
    if outputs_hook is None and any(is_untyped(out.dtype) for out in signature.interface.outputs):
        raise NodeDeclarationError(f"{function.__name__} declares an output typed Any but no compute_outputs to type it from what arrives", code="any_output_without_hook")
    open_input_template = None
    if inputs_hook is not None:
        if signature.var_keyword_name_and_hint is None:
            raise NodeDeclarationError(f"{function.__name__} has a compute_added_inputs hook but no **parameter for its inputs to land on", code="compute_added_inputs_without_var_keyword")
        if any(isinstance(extra, _AsItArrivesMark) for extra in getattr(signature.var_keyword_name_and_hint[1], "__metadata__", ())):
            raise NodeDeclarationError(f"**{signature.var_keyword_name_and_hint[0]}: AsItArrives is every edge an input, so a compute_added_inputs hook has nothing to land on it", code="as_it_arrives_beside_compute_added_inputs")
    elif signature.var_keyword_name_and_hint is not None:
        open_input_template = input_template_for_var_keyword(*signature.var_keyword_name_and_hint)
    return NodeVersion(
        body=function, interface=signature.interface, policy=policy, deprecation=deprecation, from_run=signature.from_run, return_shape=signature.return_shape,
        open_input_template=open_input_template, compute_added_inputs=inputs_hook, compute_outputs=outputs_hook, upgrade_from_previous=upgrade,
    )


def _raise_if_upgrade_is_missing(older: NodeVersion, newer: NodeVersion) -> None:
    """A field name of the replaced version that is gone from the newer one needs an upgrade to say where it went."""
    gone_inputs = {i.name for i in older.interface.inputs} - {i.name for i in newer.interface.inputs}
    gone_outputs = {o.name for o in older.interface.outputs} - {o.name for o in newer.interface.outputs}
    upgrade = newer.upgrade_from_previous
    if upgrade is None:
        unexplained = gone_inputs | gone_outputs
    elif upgrade.rewrite is None:
        unexplained = (gone_inputs - set(upgrade.renamed_inputs)) | (gone_outputs - set(upgrade.renamed_outputs))
    else:
        unexplained = set()
    if unexplained:
        raise NodeDeclarationError(f"the version it replaces has {sorted(unexplained)}, gone from this one; its Upgrade says where they went", code="upgrade_missing")


@dataclass(frozen=True, eq=False, repr=False)
class NodeDefinition(Generic[_P, _R]):
    """A node: what ``@node`` returns, what a registry holds and what a graph names by ``id``.

    Frozen, and callable with the function's own signature, so a node is
    tested by calling it. ``versions`` holds one record per version, keyed
    by number, and ``current_version`` is the highest.
    """

    #: The registry id. Stored in every graph that uses this node, so
    #: changing it is a data migration, not a rename.
    id: str
    #: What a person sees in an editor's list of nodes. A node copies these when it is
    #: added to a graph and may edit its own copy.
    title: str
    description: str
    #: Where an editor lists it. A plain string; the host keeps the table
    #: that titles categories.
    category: str
    tags: tuple[str, ...]
    #: Long-form markdown for the node's help. ``None`` means the
    #: description is all there is to say.
    docs: str | None
    versions: Mapping[int, NodeVersion]
    current_version: int
    #: The host's own data about the node, as JSON: kept and described,
    #: never read by conductor.
    extra: Mapping[str, JsonValue]

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Call the newest version's function, as a test or a host would."""
        body = self.versions[self.current_version].body
        if not callable(body):
            raise TypeError(f"{self.id!r} is a graph, so it has no function to call; run it through a compiled graph")
        return body(*args, **kwargs)

    def graph_node(self, id: str, /, *args: _P.args, **kwargs: _P.kwargs) -> GraphNode:
        """This node as it sits in a graph, at its current version, called the way its function is::

            upper.graph_node("b", text=From("a.result"), times=2)

        A value becomes a static; ``From`` is an edge. ``OPEN`` leaves an input with
        nothing on it, for the caller to fill or a ``FromRun`` to supply.
        """
        body = self.versions[self.current_version].body
        if callable(body):
            try:
                bound = inspect.signature(body).bind_partial(*args, **kwargs).arguments
            except TypeError as wrong:
                inputs = ", ".join(input_.name for input_ in self.versions[self.current_version].interface.inputs)
                raise TypeError(f"{self.id!r} placed as {id!r}: {wrong}; its inputs are {inputs or 'none'}") from wrong
            spread = next((name for name, parameter in inspect.signature(body).parameters.items() if parameter.kind is inspect.Parameter.VAR_KEYWORD), None)
            values = {**{name: value for name, value in bound.items() if name != spread}, **bound.get(spread, {})}
        elif args:
            raise TypeError(f"{self.id!r} is a graph; its inputs are given by name")
        else:
            values = kwargs
        bindings = {name: value if isinstance(value, (From, Static)) else Static(value) for name, value in values.items() if value is not OPEN}
        return GraphNode(id=id, type=self.id, version=self.current_version, bindings=bindings)

    def __repr__(self) -> str:
        """The record as the call that states what it is, optional parts only when set."""
        parts = [f"id={self.id!r}", f"title={self.title!r}", f"category={self.category!r}"]
        if self.tags:
            parts.append(f"tags={self.tags!r}")
        parts.append(f"versions={tuple(sorted(self.versions))!r}")
        return f"NodeDefinition({', '.join(parts)})"

    @classmethod
    def from_compiled_graphs(
        cls,
        id: str,
        *,
        title: str | None = None,
        description: str = "",
        category: str = "",
        tags: Sequence[str] = (),
        extra: Mapping[str, JsonValue] | None = None,
        docs: str | None = None,
        deprecation: Deprecation | None = None,
        versions: CompiledGraph | Mapping[int, CompiledGraph],
    ) -> NodeDefinition[..., Any]:
        """A node whose versions are graphs a host compiled, by number; its numbers may have holes. One graph alone is
        version 1. The title defaults to the id, as ``@node``'s does to the function's name. ``deprecation`` is the newest version's."""
        if not isinstance(versions, Mapping):
            versions = {1: versions}
        if not versions:
            raise NodeDeclarationError(f"{id!r} has no versions; a definition declares at least one", code="versions_empty")
        return NodeDefinition(
            id=id, title=title or humanised(id), description=description, category=category, tags=tuple(tags), docs=docs,
            versions=MappingProxyType({
                number: NodeVersion(body=graph, interface=graph._embeddable.interface, policy=None, deprecation=deprecation if number == max(versions) else None)
                for number, graph in versions.items()
            }),
            current_version=max(versions),
            extra=MappingProxyType(checked_extra(extra)),
        )

    def describe(self) -> NodeDescription:
        """This definition as a ``NodeDescription``: what an editor reads."""
        return NodeDescription(
            id=self.id, title=self.title, description=self.description, category=self.category, tags=self.tags,
            extra=dict(self.extra), docs=self.docs,
            versions={
                number: VersionDescription(
                    interface=version.interface,
                    policy=version.policy,
                    open_input_type=None if version.open_input_template is None else version.open_input_template.dtype,
                    deprecation=version.deprecation,
                )
                for number, version in self.versions.items()
            },
            current_version=self.current_version,
        )


def node(
    id: str,
    /,
    *,
    title: str | None = None,
    description: str | None = None,
    category: str = "",
    tags: Sequence[str] = (),
    extra: Mapping[str, JsonValue] | None = None,
    docs: str | None = None,
    policy: Policy = Policy(),
    deprecation: Deprecation | None = None,
    compute_added_inputs: Callable[..., Iterable[Input]] | None = None,
    compute_outputs: Callable[..., Iterable[Output]] | None = None,
    replaces: NodeDefinition[..., Any] | None = None,
    upgrade: Upgrade | None = None,
) -> Callable[[Callable[_P, _R]], NodeDefinition[_P, _R]]:
    """Declare a node from the function below it (see the module docstring)."""
    if not isinstance(id, str) or not id:
        raise NodeDeclarationError(f"@node takes the node's id first, a non-empty string: @node('upper'); got {id!r}", code="node_id_invalid")

    def decorate(function: Callable[_P, _R]) -> NodeDefinition[_P, _R]:
        try:
            if upgrade is not None and replaces is None:
                raise NodeDeclarationError("has an upgrade but replaces nothing; name the version it upgrades with replaces=", code="upgrade_without_replaces")
            if replaces is not None and (not isinstance(replaces, NodeDefinition) or replaces.id != id):
                raise NodeDeclarationError(f"replaces {replaces!r}; a node replaces an earlier @node of its own id", code="replaces_another_node")
            checked = checked_extra(extra)
            version = _version_of_function(function, policy, deprecation, compute_added_inputs, compute_outputs, upgrade)
            if replaces is not None:
                _raise_if_upgrade_is_missing(replaces.versions[replaces.current_version], version)
        except NodeDeclarationError as error:
            raise NodeDeclarationError(f"@node({id!r}) on {function.__name__}: {error}", code=error.code) from error
        number = 1 if replaces is None else replaces.current_version + 1
        return NodeDefinition(
            id=id,
            title=title or humanised(function.__name__),
            description=(inspect.getdoc(function) or "").split("\n\n")[0].replace("\n", " ") if description is None else description,
            category=category,
            tags=tuple(tags),
            docs=docs,
            versions=MappingProxyType({**({} if replaces is None else replaces.versions), number: version}),
            current_version=number,
            extra=MappingProxyType(checked),
        )

    return decorate


_EXTRA = TypeAdapter(dict[str, JsonValue])


def checked_extra(extra: Mapping[str, JsonValue] | None) -> dict[str, JsonValue]:
    """``extra`` as a plain dict, once it is JSON; anything else raises."""
    if extra is None:
        return {}
    try:
        return _EXTRA.validate_python(dict(extra), strict=True)
    except (ValidationError, TypeError) as wrong:
        raise NodeDeclarationError(f"extra must be JSON, a dict of str to JSON values; got {extra!r}", code="extra_not_json") from wrong


class VersionDescription(ConductorModel):
    """One version as an editor reads it: its interface, its policy, what an edge
    into it becomes when its inputs are open, and its deprecation notice.

    ``policy`` is ``None`` for a version whose body is a graph, which
    nothing runs as one unit.
    """

    interface: Interface
    policy: Policy | None
    open_input_type: DTypeField | None
    deprecation: Deprecation | None


class NodeDescription(ConductorModel):
    """A node definition as a record, what an editor lists.

    Built by ``NodeDefinition.describe()`` on demand and read by an editor:
    what it offers is these records dumped through pydantic. Describes the
    *type*. The titles one node in a graph shows live on its ``GraphNode``.
    """

    id: str
    title: str
    description: str
    category: str
    tags: tuple[str, ...]
    extra: dict[str, JsonValue]
    docs: str | None
    versions: dict[int, VersionDescription]
    current_version: int
