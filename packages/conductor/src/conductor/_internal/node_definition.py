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

One output is ``result``; a frozen dataclass returned gives one output per
field. ``Series[X]`` and ``Table[R]`` give many rows, ``SKIPPED`` on an output
is a branch not taken, and ``raise Ask(...)`` stops the row for a person.

A newer version of a node names the one it ``replaces``, itself a ``@node``
of the same id, and says with ``upgrade`` how a node saved at that one moves
here. A definition is one version; the older ones hang off ``replaces``::

    @node("split")
    def split_v1(text: str, sep: str = "\\n") -> Series[str]: ...

    @node("split", replaces=split_v1, upgrade=Upgrade(renamed_inputs={"sep": "separator"}))
    def split(text: str, separator: str = "\\n") -> Series[str]: ...

Two hooks shape a node from what it is given in one graph, each a plain
function run when a graph is loaded, never per row. A parameter of a hook
names an input of the node and receives the value typed into it (or its
default); a parameter annotated ``type`` receives the type arriving on the
input of that name by its edge (``Any`` until one does). A hook returns
``{name: type hint}``, read like a signature. ``compute_added_inputs`` adds
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Generic, ParamSpec, Protocol, TypeVar, overload

from pydantic import JsonValue, TypeAdapter, ValidationError

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import NodeDeclarationError
from conductor._internal.fields import (
    REQUIRED,
    SKIPPED,
    FieldSpec,
    Param,
    _FromRunMarker,
    humanised,
    read_field,
)
from conductor._internal.record import GraphRecord

_P = ParamSpec("_P")
_R = TypeVar("_R")


@dataclass(frozen=True)
class Policy:
    """How the engine calls a node, set by its author: ``@node("fetch", policy=Policy(retries=3))``.

    ``concurrency`` rows are called at once, on threads; with 1 and no
    ``attempt_timeout`` the node is called in order, on the leg's own thread.
    A call that raises ``ExternalFailure``, or one of the ``retry_on`` classes
    of the node's own client, is tried ``retries`` more times, the first after
    ``retry_delay`` seconds and each later one twice as long; anything else
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

    def describe(self) -> dict[str, Any]:
        """The policy as JSON; ``retry_on`` is classes, which have no JSON form, so it is left out."""
        return {"concurrency": self.concurrency, "retries": self.retries, "retry_delay": self.retry_delay,
                "attempt_timeout": self.attempt_timeout, "reuse_when_inputs_match": self.reuse_when_inputs_match}


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

    def version_of(self, version: int) -> NodeDefinition[..., Any] | None:
        """This node at ``version``, walking back through what each version replaces; ``None`` when there is none."""
        found: NodeDefinition[..., Any] | None = self
        while found is not None and found.version != version:
            found = found.replaces
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

    def describe(self) -> dict[str, Any]:
        """The node as an editor's palette shows it, as JSON."""
        return {
            "id": self.id,
            "version": self.version,
            "versions": list(self.versions),
            "title": self.title,
            "description": self.description,
            "category": self.category,
            "tags": list(self.tags),
            "extra": dict(self.extra),
            "docs": self.docs,
            "deprecation": None if self.deprecation is None else self.deprecation.model_dump(mode="json"),
            "is_graph": self.graph is not None,
            "policy": self.policy.describe(),
            "inputs": [f.describe() for f in self.inputs.values()],
            "outputs": [f.describe() for f in self.outputs.values()],
            "takes_added_inputs": self.compute_added_inputs is not None,
            "computes_outputs": self.compute_outputs is not None,
        }

    def spread(self, returned: Any) -> Any:
        """What the function returned, as ``{output: value}``; ``SKIPPED`` when the whole call was a branch not taken."""
        if returned is SKIPPED:
            return SKIPPED
        if not self.outputs:
            return {}
        if self.compute_outputs is not None:
            return {name: returned[name] for name in self.outputs}
        if dataclasses.is_dataclass(self.returns):
            return {name: getattr(returned, name) for name in self.outputs}
        return {"result": returned}


class _Declare(Protocol):
    """What ``@node(...)`` returns: it makes a node of a function, or of a stored graph."""

    @overload
    def __call__(self, target: GraphRecord, /) -> NodeDefinition[..., Any]: ...

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
) -> _Declare:
    """Make the function below it, or a stored graph, a node (see the module docstring).

    ``version`` is one past the version it ``replaces``, or 1, unless given:
    a host whose stored graphs have their own numbers names them.
    """
    if not isinstance(id, str) or not id:
        raise NodeDeclarationError(f"@node takes the node's id first, a non-empty string: @node('upper'); got {id!r}", code="node_id_invalid")
    if upgrade is not None and replaces is None:
        raise NodeDeclarationError(f"@node({id!r}) has an upgrade but replaces nothing; name the version it upgrades with replaces=", code="upgrade_without_replaces")
    if replaces is not None and (not isinstance(replaces, NodeDefinition) or replaces.id != id):
        raise NodeDeclarationError(f"@node({id!r}) replaces {replaces!r}; a node replaces an earlier @node of its own id", code="replaces_another_node")
    number = version if version is not None else (replaces.version + 1 if replaces is not None else 1)
    if replaces is not None and number <= replaces.version:
        raise NodeDeclarationError(f"@node({id!r}) is version {number}, not after the version {replaces.version} it replaces", code="version_not_after")
    checked = MappingProxyType(checked_extra(extra))

    def make(target: Any) -> NodeDefinition[..., Any]:
        common: dict[str, Any] = {
            "id": id, "version": number, "category": category, "tags": tuple(tags), "extra": checked, "docs": docs,
            "deprecation": deprecation, "replaces": replaces, "upgrade": upgrade, "policy": policy or Policy(),
        }
        if not callable(target):
            record = target if isinstance(target, GraphRecord) else GraphRecord.model_validate(target)
            return NodeDefinition(function=None, inputs={}, outputs={}, graph=record, title=title or humanised(id), description=description or "", **common)
        where = f"@node({id!r}) on {getattr(target, '__name__', target)}: "
        if inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target):
            raise NodeDeclarationError(f"{where}a node is a plain def; the engine gives a slow one threads with Policy(concurrency=)", code="async_run")
        inputs, outputs, returns = _read_signature(where, target, compute_outputs is not None)
        from_run = {name: typing.get_args(spec.type)[0] for name, spec in inputs.items() if spec.param is _FROM_RUN}
        inputs = {name: spec for name, spec in inputs.items() if name not in from_run}
        takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in inspect.signature(target).parameters.values())
        if compute_added_inputs is not None and not takes_kwargs:
            raise NodeDeclarationError(f"{where}a node with compute_added_inputs takes them as **kwargs", code="compute_added_inputs_without_var_keyword")
        if compute_added_inputs is None and takes_kwargs:
            raise NodeDeclarationError(f"{where}**kwargs takes the inputs compute_added_inputs adds; the node has no such hook", code="var_keyword_without_hook")
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

    return typing.cast(_Declare, make)


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
            raise NodeDeclarationError(f"{where}{name}: a node's inputs are passed by name", code="parameter_not_by_name")
        if name not in hints:
            raise NodeDeclarationError(f"{where}{name}: an input needs a type hint", code="input_without_type")
        default = REQUIRED if parameter.default is inspect.Parameter.empty else parameter.default
        hint = hints[name]
        if typing.get_origin(hint) is typing.Annotated and any(isinstance(m, _FromRunMarker) for m in hint.__metadata__):
            inputs[name] = FieldSpec(name, "one", hint, param=_FROM_RUN)
            continue
        inputs[name] = read_field(name, hint, default, where)
        if inputs[name].param.exclusive_group is not None:
            raise NodeDeclarationError(f"{where}{name}: exclusive_group is for outputs", code="exclusive_group_on_input")
    if "return" not in hints:
        raise NodeDeclarationError(f"{where}a node needs a return type", code="output_without_type")
    returns = hints["return"]
    if computed_outputs:
        return inputs, {}, returns
    if returns is None or returns is type(None):
        return inputs, {}, None
    if isinstance(returns, type) and dataclasses.is_dataclass(returns):
        if not returns.__dataclass_params__.frozen:  # type: ignore[attr-defined]
            raise NodeDeclarationError(f"{where}a dataclass of outputs is frozen", code="outputs_not_frozen")
        own = typing.get_type_hints(returns, include_extras=True)
        return inputs, {f.name: read_field(f.name, own[f.name], where=where) for f in dataclasses.fields(returns)}, returns
    if typing.get_origin(returns) in (dict, Mapping) or returns in (dict, Mapping, Any):
        raise NodeDeclarationError(
            f"{where}return one type, or a frozen dataclass for several outputs, not {returns!r}; outputs that depend on the graph come from compute_outputs",
            code="mapping_return_without_hook",
        )
    return inputs, {"result": read_field("result", returns, where=where)}, returns


def _check_hook(where: str, hook: Callable[..., Any], kind: str, inputs: Mapping[str, FieldSpec]) -> None:
    """A hook's parameters each name an input of the node, by name."""
    for name, parameter in inspect.signature(hook).parameters.items():
        arriving = parameter.annotation in (type, "type")
        if parameter.kind is not parameter.POSITIONAL_OR_KEYWORD and parameter.kind is not parameter.KEYWORD_ONLY:
            raise NodeDeclarationError(f"{where}{kind} parameter {name!r}: a hook's parameters are passed by name", code="hook_parameter_unknown")
        if arriving and kind == "compute_added_inputs":
            raise NodeDeclarationError(f"{where}compute_added_inputs parameter {name!r} reads an arriving type, but inputs are decided before anything arrives", code="arriving_on_compute_added_inputs")
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
