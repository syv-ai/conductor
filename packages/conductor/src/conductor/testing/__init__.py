"""``check_registry``: one test for every node in a registry.

``@node`` checks a node when it is declared, but some faults only show when the node is used. A hook may fail on
the node's own defaults. An older version may not upgrade cleanly to the newest. A node may change a list it was
given, which other calls of the run also see. ``check_registry`` looks for all of these, so a node package or a
host adds one test to its suite::

    from conductor.testing import check_registry

    def test_nodes():
        assert check_registry(registry, examples={"upper": [{"text": "hi"}]}) == []

A node's function is only called for a node you give an example for, since a node might send mail or charge a card.
Each example runs through the engine, as a graph of that one node with the example's values typed in.

Each problem it finds is a ``Problem``, and its ``loc`` is ``(node_id, version)``. Here is what each code means:

- ``hook_failed``: a ``compute_added_inputs`` or ``compute_outputs`` hook fails when it is called with the node's
  defaults and made-up values. ``example_hook_failed``: it fails on an example.
- ``needs_example``: a check that needs no example, of the node's hooks or of an older version's upgrade, can't
  make up a value for a required input that takes only a constant.
- ``subgraph_broken``: a subgraph's own graph doesn't compile on the registry's nodes, so ``registry.describe()``
  leaves it out.
- ``upgrade_broken``: ``registry.upgrade`` refuses to move a graph saved with an older version of the node to the
  newest, or the graph it returns doesn't compile.
- ``changed_its_inputs``: the node changed a value it was given.
- ``example_run_failed``: an example didn't run to the end, or to a pause for a person. ``example_not_compiled``: its graph
  of one node doesn't compile. ``example_refused``: running it is refused, as for a missing context.
- ``example_for_no_node``: an example names a node or a version that isn't in the registry. Its ``loc`` is
  ``(node_id,)`` when the example gives no version.

These codes are ``check_registry``'s own, in ``REGISTRY_CHECK_PROBLEMS``, and aren't in ``PROBLEMS``. A developer sees
them when a test fails, and a host never meets them. Each one, with an example, is on the problems page: ``python -m conductor.about problems``.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import typing
from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

import polars as pl
from pydantic import ValidationError
from typing_extensions import sentinel

from conductor._internal.base import REQUIRED
from conductor._internal.field import Field
from conductor._internal.graph import Graph
from conductor._internal.node import (
    FunctionDefinition,
    NodeDefinition,
    SubgraphDefinition,
)
from conductor._internal.problem import InvalidCallError, Problem, compile_problems
from conductor._internal.record import GraphRecord, InputEntry, NodeRecord, WrittenValuesValidator
from conductor._internal.registry import NodeRegistry
from conductor._internal.types import Param

from . import problems
from .problems import REGISTRY_CHECK_PROBLEMS

#: What ``_made_up`` returns for a field it can't make up a value for.
_CANNOT = sentinel("_CANNOT")

#: The codes of a ``compute_...`` hook that failed: it raised a ``NodeError`` or returned what compiling can't use,
#: or it raised any other exception.
_HOOK_FAILED = frozenset(
    {compile_problems.compute_hook_failed.code, compile_problems.compute_hook_raised.code}
)


def check_registry[C](
    registry: NodeRegistry[C],
    /,
    *,
    examples: Mapping[str | tuple[str, int], Sequence[Mapping[str, Any]]] = MappingProxyType({}),
    context: C | None = None,
) -> list[Problem]:
    """Checks every node of a registry, and returns every problem it finds.

    It doesn't stop at the first problem, so one test run shows them all. It returns ``[]`` when every node is
    fine. Every version of every node is checked, the older ones too.

    Args:
        registry: The registry to check.
        examples: Calls to make, by node id. Each call is a mapping from input name to value, such as
            ``{"upper": [{"text": "hi"}]}``. For an older version of a node, use ``(id, version)`` as the key. A
            node with no example has its hooks checked, but its function is never called.
        context: The run's context, passed to each example's run the way ``graph.run(context=...)`` takes it.
            It is needed when the registry's nodes take ``Context[X]``.

    Returns:
        The problems found. Those about ``examples`` come first, then each node's, in the registry's order.
    """
    found: list[Problem] = []
    by_version: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for key, calls in examples.items():
        if isinstance(key, str):  # a key with only an id is for the newest version
            if key not in registry:
                found.append(problems.example_for_no_node(key, named=repr(key)))
                continue
            place = (key, registry[key].version)
        elif key[0] not in registry or key[1] not in registry[key[0]].versions:
            found.append(problems.example_for_no_node(*key, named=repr(key)))
            continue
        else:
            place = key
        by_version.setdefault(place, []).extend(calls)
    for definition in registry:
        found += _problems_of(definition, registry, by_version, context)
    return found


def _problems_of(
    definition: NodeDefinition,
    registry: NodeRegistry[Any],
    examples: Mapping[tuple[str, int], Sequence[Mapping[str, Any]]],
    context: Any,
) -> list[Problem]:
    """Checks one node: its newest version, then each older one and its upgrade to the newest.

    An older version with no example is upgraded with made-up values. When a required input that takes only a
    constant gets none, the version is ``needs_example``, and neither its upgrade nor its hooks are checked, since
    both need that value.
    """
    found = _version_problems(
        definition, registry, examples.get((definition.id, definition.version), ()), context
    )
    for older in list(definition.versions.values())[1:]:
        given = examples.get((older.id, older.version), ())
        if given:
            values = dict(given[0])
        else:
            match older:
                case FunctionDefinition(inputs=inputs):
                    made = _made_up_values(inputs)
                case SubgraphDefinition():  # its inputs' types aren't known until a graph that uses it is compiled
                    made = {}
            if isinstance(made, list):
                found.append(problems.needs_example(older.id, older.version, inputs=made))
                continue
            values = made
        found += _upgrade_problems(older, registry, values)
        found += _version_problems(older, registry, given, context)
    return found


def _version_problems(
    version: NodeDefinition,
    registry: NodeRegistry[Any],
    given: Sequence[Mapping[str, Any]],
    context: Any,
) -> list[Problem]:
    """Checks one version of a node.

    A subgraph's own graph is compiled first, as it is. One with problems is ``subgraph_broken``, and its examples
    aren't run, since each would fail on the same problems. Then, with no examples, only a function's hooks are
    checked. With examples, each one is run.
    """
    if isinstance(version, SubgraphDefinition):
        graph_problems = Graph.compile(version.graph, registry).problems
        if graph_problems:
            return [
                problems.subgraph_broken(
                    version.id,
                    version.version,
                    reason=_reason(graph_problems),
                    problems=[p.model_dump(mode="json") for p in graph_problems],
                )
            ]
    if not given:
        return _hook_problems(version, registry)
    return [
        problem
        for example in given
        for problem in _run_problems(version, registry, example, context)
    ]


def _one_node_graph(
    version: NodeDefinition, constants: Mapping[str, Any], graph_inputs: Iterable[str] = ()
) -> GraphRecord:
    """Writes a graph record with just this node in it, under its own id and at its version.

    Args:
        version: The node, at the version to use.
        constants: The values to type into its inputs, by input name.
        graph_inputs: The inputs to make graph inputs, each under the input's own name, unless ``constants`` gives
            it a value.

    Raises:
        InvalidCallError: ``invalid_value`` at the node and input, for a value with no JSON form.
    """
    inputs = {
        name: InputEntry(graph_input=name) for name in graph_inputs if name not in constants
    } | {
        name: InputEntry(constant=value)
        for name, value in WrittenValuesValidator(version.id)(constants).items()
    }
    return GraphRecord(
        nodes={version.id: NodeRecord(node=version.id, version=version.version, inputs=inputs)}
    )


def _hook_problems(version: NodeDefinition, registry: NodeRegistry[Any]) -> list[Problem]:
    """Checks a node's hooks without an example.

    The node is compiled in a graph of its own, which calls its hooks. Its inputs keep their defaults. Each
    required input that takes only a constant gets a made-up value, so the hooks have a value to read.
    """
    match version:
        case FunctionDefinition(hook_parameters=hooks) if hooks:
            pass
        case (
            FunctionDefinition() | SubgraphDefinition()
        ):  # a function with no hooks, or a subgraph, which never has hooks of its own
            return []
    values = _made_up_values(
        {
            name: field
            for name, field in version.inputs.items()
            if field.param.constant_only and field.default is REQUIRED
        }
    )
    if isinstance(values, list):
        return [problems.needs_example(version.id, version.version, inputs=values)]
    failed = [
        p
        for p in Graph.compile(_one_node_graph(version, values), registry).problems
        if p.type in _HOOK_FAILED
    ]
    return [
        problems.hook_failed(
            version.id, version.version, reason=_reason([p]), problems=[p.model_dump(mode="json")]
        )
        for p in failed
    ]


def _upgrade_problems(
    older: NodeDefinition, registry: NodeRegistry[Any], values: Mapping[str, Any]
) -> list[Problem]:
    """Checks that an older version of a node upgrades cleanly to the newest.

    The node is put in a graph at the older version, with ``values``, an example's or made-up ones, as constants.
    Each required input that gets no value, such as one of type ``T`` that no value can be made up for, is a graph
    input. Then ``registry.upgrade`` moves it to the newest version, and the result is compiled.
    """
    newest = registry[older.id].version
    try:
        record = _one_node_graph(older, values, older.required_inputs)
    except InvalidCallError:
        return []  # the example's own run reports the values that can't be typed in
    try:
        upgraded = registry.upgrade(record, older.id)
    except InvalidCallError as refused:
        behind = refused.problems
    else:
        behind = Graph.compile(upgraded, registry).problems
    if not behind:
        return []
    return [
        problems.upgrade_broken(
            older.id,
            older.version,
            newest=newest,
            reason=_reason(behind),
            problems=[p.model_dump(mode="json") for p in behind],
        )
    ]


def _run_problems(
    version: NodeDefinition, registry: NodeRegistry[Any], example: Mapping[str, Any], context: Any
) -> list[Problem]:
    """Runs one example of a node through the engine, and returns what went wrong."""
    match version:
        case SubgraphDefinition():  # a subgraph needs the registry's other nodes to run
            definitions, changed = registry.filter(lambda held: held.id != version.id), list[str]()
            definitions.add(version)
        case FunctionDefinition():
            watched, changed = _watched(version)
            definitions = registry.filter(lambda held: held.id == version.id)
            definitions.replace(watched)
    try:
        record = _one_node_graph(version, dict(example))
    except InvalidCallError as refused:
        return _not_compiled(version, refused.problems)
    graph = Graph.compile(record, definitions)
    if graph.problems:
        return _not_compiled(version, graph.problems)
    try:
        run = graph.run(context=context)
    except InvalidCallError as refused:
        call_problems = refused.problems
        return [
            problems.example_refused(
                version.id,
                version.version,
                reason=_reason(call_problems),
                problems=[p.model_dump(mode="json") for p in call_problems],
            )
        ]
    found = [
        problems.changed_its_inputs(version.id, version.version, input=name) for name in changed
    ]
    if (cause := run.cause) is not None:
        found.append(
            problems.example_run_failed(
                version.id,
                version.version,
                status=run.status,
                reason=_reason([cause]),
                problems=[cause.model_dump(mode="json")],
            )
        )
    return found


def _not_compiled(version: NodeDefinition, graph_problems: Sequence[Problem]) -> list[Problem]:
    """Returns the one problem for an example that doesn't compile.

    It is ``example_hook_failed`` when a hook caused every problem, and ``example_not_compiled`` otherwise.
    """
    reason = _reason(graph_problems)
    written = [p.model_dump(mode="json") for p in graph_problems]
    if all(p.type in _HOOK_FAILED for p in graph_problems):
        return [
            problems.example_hook_failed(
                version.id, version.version, reason=reason, problems=written
            )
        ]
    return [
        problems.example_not_compiled(version.id, version.version, reason=reason, problems=written)
    ]


def _reason(behind: Sequence[Problem]) -> str:
    """Returns the ``reason`` of a ``check_registry`` problem: the messages of the problems behind it.

    The messages are joined by ``"; "``. An exception the node raised is an ``execution_failed`` problem, whose
    message doesn't say which exception it was. So its class name is used instead, as in
    ``"the node raised ZeroDivisionError"``.
    """
    return "; ".join(
        f"the node raised {p.ctx['error']}" if p.type == "execution_failed" else p.msg
        for p in behind
    )


def _watched(
    version: FunctionDefinition[..., Any],
) -> tuple[FunctionDefinition[..., Any], list[str]]:
    """Wraps a node's function to notice when a call changes an input in place.

    The wrapped node is called just as before, but it is never retried. The run's context, which a node that takes
    it gets first, is passed on and isn't watched.

    Returns:
        The wrapped node, and the list it adds the name of each changed input to.
    """
    function = version.function
    changed: list[str] = []

    def watched(*context: Any, **kwargs: Any) -> Any:
        before: dict[str, object] = {}
        for name, value in kwargs.items():
            try:
                kept = copy.deepcopy(value)
            except Exception:  # noqa: BLE001, S112  a value that can't be copied can't be checked
                continue
            if _same(
                kept, value
            ):  # a value that isn't equal to its own copy, like NaN or an object without __eq__, can't be checked
                before[name] = kept
        try:
            return function(*context, **kwargs)
        finally:
            changed.extend(
                name
                for name, value in before.items()
                if name not in changed and not _same(value, kwargs[name])
            )

    return dataclasses.replace(
        version, function=watched, policy=dataclasses.replace(version.policy, retries=0)
    ), changed


def _made_up_values(fields: Mapping[str, Field]) -> dict[str, Any] | list[str]:
    """Makes up a value for each field that one can be made up for.

    Returns:
        The values, by name. Or, when a required field that takes only a constant gets none, the names of every
        such field, sorted: only a constant can give it a value, so the node needs an example.
    """
    values = {
        name: made for name, field in fields.items() if (made := _made_up(field)) is not _CANNOT
    }
    missing = sorted(
        name
        for name, field in fields.items()
        if name not in values and field.param.constant_only and field.default is REQUIRED
    )
    if missing:
        return missing
    return values


def _made_up(field: Field) -> Any:
    """Makes up a value for a field, within its limits.

    Returns:
        The value, as the field's type reads it, or ``_CANNOT`` when no value can be made up.
    """
    read = field.whole.argument_validator.validate_python
    for item in _candidates(
        field.type.options[0] if len(field.type.options) == 1 else None, field.param
    ):
        try:
            return read([item, item] if field.is_series else item)
        # the value is outside a limit, or doesn't match a pattern, so try the next one
        except ValidationError:
            continue
    return _CANNOT


def _candidates(hint: Any, param: Param) -> list[Any]:
    """Returns values a field of type ``hint`` might take, to be tried in order.

    The field's own type checks each one, and the first it accepts is used.
    """
    if typing.get_origin(hint) is Literal:
        return [typing.get_args(hint)[0]]
    if not isinstance(hint, type):
        return []
    if issubclass(hint, bool):
        return [True]
    if issubclass(hint, (int, float)):
        # the limits are numbers here, since a Param with other limits on a number is refused
        low: Any = param.ge if param.ge is not None else param.gt
        high: Any = param.le if param.le is not None else param.lt
        middle: list[Any] = []
        if low is not None and high is not None:
            middle = [(low + high) // 2] if issubclass(hint, int) else [(low + high) / 2]
        near = [
            *(() if low is None else (low, low + 1)),
            *(() if high is None else (high, high - 1)),
        ]
        return [*middle, *near, 1, 0, -1]
    if issubclass(hint, str):
        return ["x" * max(1, param.min_length or 1)]
    if issubclass(hint, datetime.datetime):
        return [datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)]
    if issubclass(hint, datetime.date):
        return [datetime.date(2026, 1, 1)]
    return []


def _same(one: Any, other: Any) -> bool:
    """Whether two values are equal, comparing dicts and lists item by item, and frames cell by cell.

    A frame is the same only as another frame with equal cells. A list of records with the frame's rows is not the
    same: a node that swaps one for the other has changed what the next call of the run gets. ``==`` can't be used
    on a frame, since it returns a frame of results, or raises when the other side is not a frame. For any other
    value where ``==`` raises, the two are equal only when they are the same object.
    """
    if one is other:
        return True
    if isinstance(one, Mapping) and isinstance(other, Mapping):
        return one.keys() == other.keys() and all(_same(one[key], other[key]) for key in one)
    if isinstance(one, list) and isinstance(other, list):
        return len(one) == len(other) and all(_same(a, b) for a, b in zip(one, other, strict=True))
    if isinstance(one, pl.DataFrame) or isinstance(other, pl.DataFrame):
        return (
            isinstance(one, pl.DataFrame) and isinstance(other, pl.DataFrame) and one.equals(other)
        )
    try:
        return bool(one == other)
    except (ValueError, TypeError):
        return one is other


__all__ = ["check_registry", "REGISTRY_CHECK_PROBLEMS"]
