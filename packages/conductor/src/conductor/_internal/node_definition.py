"""``@node``: a plain function, or a stored graph, becomes a frozen ``NodeDefinition``.

A node is a function. Its typed signature is its interface::

    @node("upper")
    def upper(text: str) -> str:
        \"\"\"Upper-cases a text.\"\"\"
        return text.upper()

``@node`` wraps the function into a value and registers nothing, so
``upper("a")`` is the unit test and a host decides what to offer by adding
definitions to a ``NodeRegistry`` (whose ``@registry.node`` takes the same
arguments and adds the node too). The title defaults to the function's name
in everyday words and the description to its docstring's first paragraph.
``extra`` is the host's own JSON about the node, which conductor keeps and
never reads.

One output is ``result``, of whatever type is returned, a ``dict`` included;
a frozen dataclass returned gives one output per field. ``Series[X]`` and ``Table[R]`` give many rows, ``SKIPPED`` on an output
is a branch not taken, and ``raise Ask(...)`` stops the row for a person.

A newer version of a node names the one it ``replaces``, itself a ``@node``
of the same id, and says with ``upgrade`` how a node saved at that one moves
here. A definition is one version, and each version names the one before it in
``replaces``::

    @node("split")
    def split_v1(text: str, sep: str = "\\n") -> Series[str]: ...

    @node("split", replaces=split_v1, upgrade=Upgrade(renamed_inputs={"sep": "separator"}))
    def split(text: str, separator: str = "\\n") -> Series[str]: ...

Two hooks shape a node from what it is given in one graph, each a plain
function run when a graph is loaded, never per row. A parameter of a hook
names an input of the node and receives the value typed into it (or its
default); a parameter annotated ``type`` receives the type arriving on the
input of that name by its edge (``Any`` until one does). A hook returns
``{name: type hint}``, read like a signature, or ``{name: (type hint, default)}``
for a field with a default, as pydantic's ``create_model`` takes one. A hook
that raises ``NodeError`` gives its own message, code and ``ctx`` to the
problem. ``compute_added_inputs`` adds
inputs, which the function takes as ``**kwargs``; ``compute_outputs`` gives
every output, and the function returns them as a mapping. A hook that raises
is a problem of the graph, not an exception.

A stored graph is a node too: ``node("enrich")(record)``. Loading a graph that
uses it puts the stored graph's nodes in its place.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import typing
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, ParamSpec, Protocol, TypeVar, overload

from pydantic import JsonValue, TypeAdapter, ValidationError

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import InvalidCallError, NodeDeclarationError, make_problem
from conductor._internal.fields import (
    REQUIRED,
    FieldDescription,
    FieldSpec,
    Param,
    _FromRunMarker,
    humanised,
    read_field,
)
from conductor._internal.record import GraphRecord, as_record

_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass(frozen=True)
class Policy:
    """How the engine calls a node, set by its author: ``@node("fetch", policy=Policy(retries=3))``.

    ``concurrency`` rows are called at once, on threads; with 1 and no
    ``attempt_timeout`` the node is called in order, on the leg's own thread.
    A call that raises ``ExternalFailure``, or one of the ``retry_on`` classes
    of the node's own client, is tried ``retries`` more times, waiting from
    ``retry_delay`` seconds (the reference has the schedule); anything else
    fails at once, since repeating a computation repeats its failure. An
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
        if self.concurrency < 1 or self.retries < 0 or self.retry_delay < 0 or (self.attempt_timeout is not None and self.attempt_timeout <= 0):
            raise ValueError(f"{self}: concurrency is at least 1, retries and retry_delay at least 0, attempt_timeout above 0")
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
    """A notice that a version of a node is going away.

    Every part is optional prose: ``alternative`` names the node to use
    instead, ``migration`` explains how to move. It is content an editor
    shows; a graph that uses a deprecated node is not wrong, so this is never
    a ``Problem``.
    """

    header: str | None = None
    description: str | None = None
    alternative: str | None = None
    migration: str | None = None


@dataclass(frozen=True)
class Upgrade:
    """How a node saved at the version a newer one replaces becomes the newer one.

    Given on the newer version's ``@node``, beside ``replaces``; required only
    when a field name of the replaced version is gone from the newer one.
    Read only by ``registry.upgrade``: loading never upgrades, so a graph
    saved at an old version runs that version.

    ``renamed_inputs`` moves each input's value, default and edges to the new
    name before ``rewrite`` runs. ``rewrite`` takes what is typed into the
    node, its values and defaults as one dict, and returns it for the newer
    version: a key that was a default stays one, any other key is a value.
    ``renamed_outputs`` moves every edge that read the old name, or a column
    of it.
    """

    renamed_inputs: Mapping[str, str] = field(default_factory=dict)
    renamed_outputs: Mapping[str, str] = field(default_factory=dict)
    rewrite: Callable[[dict[str, Any]], dict[str, Any]] | None = None


@dataclass(frozen=True, eq=False)
class NodeDefinition(Generic[_P, _R]):
    """One version of a node: what it takes, what it gives, and how to call it.

    Frozen, and callable with the function's own signature, so a node is
    tested by calling it. ``graph`` is set instead of ``function`` for a node
    that is a stored graph; its fields are the graph's, worked out when a
    graph using it is loaded.
    """

    #: The id a graph names it by. Stored in every graph that uses it, so
    #: changing it is a data migration, not a rename.
    id: str
    version: int
    function: Callable[..., Any] | None
    inputs: Mapping[str, FieldSpec]
    outputs: Mapping[str, FieldSpec]
    #: The return annotation; a frozen dataclass when the node has several outputs.
    returns: Any = None
    title: str = ""
    description: str = ""
    #: Where an editor lists it; a plain string whose titles the host keeps.
    category: str = ""
    tags: tuple[str, ...] = ()
    #: The host's own JSON about the node: kept and described, never read here.
    extra: Mapping[str, JsonValue] = field(default_factory=dict)
    #: Long-form markdown for the node's help, when the description isn't all there is to say.
    docs: str | None = None
    deprecation: Deprecation | None = None
    replaces: NodeDefinition[..., Any] | None = None
    upgrade: Upgrade | None = None
    graph: GraphRecord | None = None
    policy: Policy = Policy()
    compute_added_inputs: Callable[..., Mapping[str, Any]] | None = None
    compute_outputs: Callable[..., Mapping[str, Any]] | None = None
    #: The ``FromRun`` parameters: the type whose value goes under each name in a call.
    from_run: Mapping[str, type] = field(default_factory=dict)

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Call the function, as a test or a host would."""
        if self.function is None:
            raise TypeError(f"{self.id!r} is a graph, so it has no function to call; load a graph that uses it")
        return typing.cast(_R, self.function(*args, **kwargs))

    def __repr__(self) -> str:
        parts = [f"id={self.id!r}", f"version={self.version}", f"title={self.title!r}"]
        if self.category:
            parts.append(f"category={self.category!r}")
        if self.tags:
            parts.append(f"tags={self.tags!r}")
        return f"NodeDefinition({', '.join(parts)})"

    def version_of(self, version: int) -> NodeDefinition[..., Any]:
        """This node at ``version``, walking back through what each version replaces; a ``KeyError`` naming the versions
        there are when it has none. ``version in definition.versions`` asks first."""
        found: NodeDefinition[..., Any] | None = self
        while found is not None and found.version != version:
            found = found.replaces
        if found is None:
            raise KeyError(f"{self.id!r} has no version {version!r}; it has {', '.join(map(str, self.versions))}")
        return found

    @property
    def versions(self) -> tuple[int, ...]:
        """Every version number, newest first."""
        found: list[int] = []
        step: NodeDefinition[..., Any] | None = self
        while step is not None:
            found.append(step.version)
            step = step.replaces
        return tuple(found)

    def describe(self) -> NodeDescription:
        """The node as an editor's palette shows it; a stored graph's fields only through ``registry.describe()``, which holds the nodes it uses."""
        return NodeDescription(
            id=self.id, version=self.version, versions=list(self.versions), title=self.title, description=self.description,
            category=self.category, tags=list(self.tags), extra=dict(self.extra), docs=self.docs, deprecation=self.deprecation,
            is_graph=self.graph is not None, policy=self.policy.describe(),
            inputs=[f.describe() for f in self.inputs.values()], outputs=[f.describe() for f in self.outputs.values()],
            takes_added_inputs=self.compute_added_inputs is not None, computes_outputs=self.compute_outputs is not None,
        )


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
    def __call__(self, target: GraphRecord | str | bytes | Mapping[str, Any], /) -> NodeDefinition[..., Any]: ...

    @overload
    def __call__(self, target: Callable[_P, _R], /) -> NodeDefinition[_P, _R]: ...

    def __call__(self, target: Any, /) -> NodeDefinition[..., Any]: ...


def node(
    id: str,
    /,
    *,
    version: int | None = None,
    title: str | None = None,
    description: str | None = None,
    category: str = "",
    tags: Sequence[str] = (),
    extra: Mapping[str, JsonValue] | None = None,
    docs: str | None = None,
    policy: Policy | None = None,
    deprecation: Deprecation | None = None,
    compute_added_inputs: Callable[..., Mapping[str, Any]] | None = None,
    compute_outputs: Callable[..., Mapping[str, Any]] | None = None,
    replaces: NodeDefinition[..., Any] | None = None,
    upgrade: Upgrade | None = None,
) -> NodeDecorator:
    """Make the function below it, or a stored graph, a node with the id ``id``::

        @node("upper")
        def upper(text: str) -> str:
            return text.upper()

    It registers nothing: ``upper("a")`` still calls the function, and
    ``registry.add(upper)`` offers it. ``@registry.node`` takes the same
    arguments and adds the node too.

    - ``title`` and ``description`` default to the function's name in
      everyday words and its docstring's first paragraph; ``docs`` is longer
      help, ``category`` and ``tags`` group it in a palette, and ``extra`` is
      the host's own JSON, kept and never read here.
    - ``policy`` says how the engine calls it (retries, a timeout, threads).
    - ``replaces`` is the version this one replaces, a ``@node`` of the same
      id, and ``upgrade`` says how a node saved at that version moves here.
      ``version`` is one past the version replaced, or 1, unless given: a host
      whose stored graphs have their own numbers names them. ``deprecation``
      marks this version going away.
    - ``compute_added_inputs`` and ``compute_outputs`` are hooks run when a
      graph is loaded, which shape the node from what is typed into it.

    A stored graph is a node too: ``node("enrich")(record)``, the record given as ``registry.load`` takes one (a
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
    if replaces is not None and (not isinstance(replaces, NodeDefinition) or replaces.id != id):
        raise NodeDeclarationError(f"@node({id!r}) replaces {replaces!r}; a node replaces an earlier @node of its own id", code="replaces_another_node")
    number = version if version is not None else (replaces.version + 1 if replaces is not None else 1)
    if replaces is not None and number <= replaces.version:
        raise NodeDeclarationError(f"@node({id!r}) is version {number}, not after the version {replaces.version} it replaces; leave version= out, or give one above {replaces.version}", code="version_not_after")
    checked = MappingProxyType(checked_extra(extra))

    def make(target: Any) -> NodeDefinition[..., Any]:
        common: dict[str, Any] = {
            "id": id, "version": number, "category": category, "tags": tuple(tags), "extra": checked, "docs": docs,
            "deprecation": deprecation, "replaces": replaces, "upgrade": upgrade, "policy": policy or Policy(),
        }
        if not callable(target):
            record = as_record(target)  # a record, JSON text or a mapping, read as registry.load reads one
            return NodeDefinition(function=None, inputs={}, outputs={}, graph=record, title=title or humanised(id), description=description or "", **common)
        where = f"@node({id!r}) on {getattr(target, '__name__', target)}: "
        if inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target):
            raise NodeDeclarationError(f"{where}a node is a plain def; the engine gives a slow one threads with Policy(concurrency=)", code="async_run")
        inputs, outputs, returns = _read_signature(where, target, compute_outputs is not None)
        from_run = {name: typing.get_args(spec.type)[0] for name, spec in inputs.items() if spec.param is _FROM_RUN}
        inputs = {name: spec for name, spec in inputs.items() if name not in from_run}
        takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in inspect.signature(target).parameters.values())
        if compute_added_inputs is not None and not takes_kwargs:
            raise NodeDeclarationError(f"{where}a node with compute_added_inputs takes them as **kwargs; add **values to its signature", code="compute_added_inputs_without_var_keyword")
        if compute_added_inputs is None and takes_kwargs:
            raise NodeDeclarationError(f"{where}**kwargs takes the inputs compute_added_inputs adds; the node has no such hook, so remove **kwargs or add compute_added_inputs=", code="var_keyword_without_hook")
        for hook, kind in ((compute_added_inputs, "compute_added_inputs"), (compute_outputs, "compute_outputs")):
            if hook is not None:
                _check_hook(where, hook, kind, inputs)
        definition: NodeDefinition[..., Any] = NodeDefinition(
            function=target, inputs=inputs, outputs=outputs, returns=returns, from_run=from_run,
            title=title or humanised(target.__name__),
            description=(inspect.getdoc(target) or "").split("\n\n")[0].replace("\n", " ") if description is None else description,
            compute_added_inputs=compute_added_inputs, compute_outputs=compute_outputs, **common,
        )
        if replaces is not None:
            _raise_if_upgrade_is_missing(where, replaces, definition)
        return definition

    return typing.cast(NodeDecorator, make)


#: What a node id never holds: each means something in an address (``node.field``, ``embedding/inner``) or a waiting id
#: (``node@row~fingerprint``).
_MARKS = frozenset("./@~")


def is_node_id(value: object) -> bool:
    """Whether ``value`` can be a node id, of a definition or of a node in a record: a non-empty text without ``. / @ ~``."""
    return isinstance(value, str) and bool(value) and not _MARKS & set(value)


def from_run_values(definitions: Iterable[tuple[str, NodeDefinition[..., Any]]], from_run: Mapping[type, Any] | None) -> dict[type, Any]:
    """The value each ``FromRun`` parameter of these placed nodes takes, by the class it names; ``InvalidCallError`` listing
    every parameter it can't fill.

    A value keyed by the class itself is taken; else the one value given that is an instance of it, so a subclass
    (a test double, a session of the host's own) is keyed by its own class or its parent's alike.
    """
    given = dict(from_run or {})
    found: dict[type, Any] = {}
    problems = []
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
            names = [getattr(key, "__name__", str(key)) for key in given]
            problems.append(make_problem("from_run_ambiguous" if fits else "from_run_missing", node_id, name, needs=needed.__name__, parameter=name, given=names))
    if problems:
        raise InvalidCallError(problems)
    return found


def _fits(needed: type, key: Any, value: Any) -> bool:
    """Whether ``value``, given under ``key``, is a ``needed``; a class no ``isinstance`` can ask about (a protocol not
    runtime-checkable) is filled only under its own key."""
    try:
        return isinstance(value, needed) or (isinstance(key, type) and issubclass(key, needed))
    except TypeError:
        return False


#: The ``Param`` a ``FromRun`` parameter is read with, so it is told apart from an input.
_FROM_RUN = Param()


def _read_signature(where: str, function: Callable[..., Any], computed_outputs: bool) -> tuple[dict[str, FieldSpec], dict[str, FieldSpec], Any]:
    """The inputs, outputs and return annotation a function's signature declares."""
    hints = typing.get_type_hints(function, include_extras=True)
    inputs = {}
    for name, parameter in inspect.signature(function).parameters.items():
        if parameter.kind is parameter.VAR_KEYWORD:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.POSITIONAL_ONLY):
            raise NodeDeclarationError(f"{where}{name}: a node's inputs are passed by name; drop the / and *args", code="parameter_not_by_name")
        if name not in hints:
            raise NodeDeclarationError(f"{where}{name}: an input needs a type hint; write one, as it is the input's interface", code="input_without_type")
        default = REQUIRED if parameter.default is inspect.Parameter.empty else parameter.default
        hint = hints[name]
        if typing.get_origin(hint) is typing.Annotated and any(isinstance(m, _FromRunMarker) for m in hint.__metadata__):
            if not isinstance(typing.get_args(hint)[0], type):
                raise NodeDeclarationError(f"{where}{name}: FromRun names the class of the value the run is given, FromRun[Session]", code="from_run_not_a_class")
            inputs[name] = FieldSpec(name, "one", hint, param=_FROM_RUN)
            continue
        inputs[name] = read_field(name, hint, default, where)
        if inputs[name].param.exclusive_group is not None:
            raise NodeDeclarationError(f"{where}{name}: exclusive_group is for outputs; put it on an output's Param", code="exclusive_group_on_input")
    if "return" not in hints:
        raise NodeDeclarationError(f"{where}a node needs a return type; write one, or -> None for no outputs", code="output_without_type")
    returns = hints["return"]
    if computed_outputs:
        return inputs, {}, returns
    if returns is None or returns is type(None):
        return inputs, {}, None
    if isinstance(returns, type) and dataclasses.is_dataclass(returns):
        if not returns.__dataclass_params__.frozen:  # type: ignore[attr-defined]
            raise NodeDeclarationError(f"{where}a dataclass of outputs is frozen; write @dataclass(frozen=True)", code="outputs_not_frozen")
        own = typing.get_type_hints(returns, include_extras=True)
        outputs = {f.name: read_field(f.name, own[f.name], _declared_default(f), where=where) for f in dataclasses.fields(returns)}
        _raise_if_setting_on(where, outputs)
        return inputs, outputs, returns
    outputs = {"result": read_field("result", returns, where=where)}
    _raise_if_setting_on(where, outputs)
    return inputs, outputs, returns


def _raise_if_setting_on(where: str, outputs: Mapping[str, FieldSpec]) -> None:
    for name, spec in outputs.items():
        if spec.param.setting:
            raise NodeDeclarationError(f"{where}{name}: a setting is an input the graph's author sets; an output is never one", code="setting_on_output")


def _declared_default(declared: dataclasses.Field[Any]) -> Any:
    """A dataclass field's default, which a person asked for the output sees proposed; ``REQUIRED`` when it has none."""
    if declared.default is not dataclasses.MISSING:
        return declared.default
    if declared.default_factory is not dataclasses.MISSING:
        return declared.default_factory()
    return REQUIRED


def _check_hook(where: str, hook: Callable[..., Any], kind: str, inputs: Mapping[str, FieldSpec]) -> None:
    """A hook's parameters each name an input of the node, by name."""
    for name, parameter in inspect.signature(hook).parameters.items():
        arriving = parameter.annotation in (type, "type")
        if parameter.kind is not parameter.POSITIONAL_OR_KEYWORD and parameter.kind is not parameter.KEYWORD_ONLY:
            raise NodeDeclarationError(f"{where}{kind} parameter {name!r}: a hook's parameters are passed by name; name each after an input of the node", code="hook_parameter_unknown")
        if arriving and kind == "compute_added_inputs":
            raise NodeDeclarationError(f"{where}compute_added_inputs parameter {name!r} reads an arriving type, but inputs are decided before anything arrives; read it in compute_outputs", code="arriving_on_compute_added_inputs")
        if not arriving and name not in inputs:
            raise NodeDeclarationError(f"{where}{kind} parameter {name!r} names no input of the node; it has {sorted(inputs)}", code="hook_parameter_unknown")


def _raise_if_upgrade_is_missing(where: str, older: NodeDefinition[..., Any], newer: NodeDefinition[..., Any]) -> None:
    """A field name of the replaced version that is gone from the newer one needs an upgrade to say where it went."""
    if older.graph is not None or newer.graph is not None:
        return
    gone_inputs = set(older.inputs) - set(newer.inputs)
    gone_outputs = set(older.outputs) - set(newer.outputs)
    upgrade = newer.upgrade
    if upgrade is None:
        unexplained = gone_inputs | gone_outputs
    elif upgrade.rewrite is None:
        unexplained = (gone_inputs - set(upgrade.renamed_inputs)) | (gone_outputs - set(upgrade.renamed_outputs))
    else:
        unexplained = set()
    if unexplained:
        raise NodeDeclarationError(f"{where}the version it replaces has {sorted(unexplained)}, gone from this one; its Upgrade says where they went", code="upgrade_missing")


_EXTRA = TypeAdapter(dict[str, JsonValue])


def checked_extra(extra: Mapping[str, JsonValue] | None) -> dict[str, JsonValue]:
    """``extra`` as a plain dict, once it is JSON: a mapping of str to JSON values, every number finite. Anything else raises."""
    if extra is None:
        return {}
    wrong_shape = NodeDeclarationError(f"extra must be JSON, a mapping of str to JSON values with finite numbers; got {extra!r}", code="extra_not_json")
    if not isinstance(extra, Mapping):
        raise wrong_shape
    try:
        checked = _EXTRA.validate_python(dict(extra), strict=True)
    except (ValidationError, TypeError, ValueError) as wrong:
        raise wrong_shape from wrong
    if not _all_finite(checked):
        raise wrong_shape
    return checked


def _all_finite(value: JsonValue) -> bool:
    """Whether no float anywhere in ``value`` is NaN or an infinity, which JSON cannot write."""
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True
