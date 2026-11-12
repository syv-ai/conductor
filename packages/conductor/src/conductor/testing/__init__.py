"""``check_registry``: one test that every node of a registry keeps what a host relies on.

A node package or a host puts one line in its test suite::

    from conductor.testing import check_registry

    def test_nodes():
        assert check_registry(registry, examples={"upper": [{"text": "hi"}]}) == []

It checks what ``@node`` can't see in a declaration. A body is called only
for a node given an example, since a node may send mail; ``examples`` maps a
node id to the inputs of its calls, by name, and ``(id, version)`` to those
of an older version. Each example runs through the engine as a graph of that
one node, its inputs typed in, given ``context`` as ``graph.run`` is: the
run's context, an instance of the registry's ``context_type``.

Every problem is a ``Problem``. Its ``loc`` is ``(node_id, version)``, except for
``example_for_no_node`` without a version, which is ``(node_id,)``:

``hook_failed``
    a ``compute_added_inputs`` or ``compute_outputs`` hook fails on the
    node's defaults, or on a value made up for an input it reads. A node with
    examples is checked on them instead.
``needs_example``
    a hook reads a required input no value can be made up for.
``upgrade_broken``
    a node saved at an older version, moved on with ``registry.upgrade``,
    doesn't load cleanly (or the upgrade raised).
``changed_its_inputs``
    the body changed a value it was given; several calls of a run can be given one.
``run_failed``
    an example didn't run to the end or to a pause.
``example_for_no_node``
    an example names a node or a version that isn't there.

A node known to fail goes in a test of its own, marked ``pytest.mark.xfail(strict=True)``,
that checks ``registry.filter(...)`` down to it. These codes are ``check_registry``'s,
not in ``PROBLEMS.codes``: a developer reads them in a test run, a host never meets them.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import typing
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any, Literal

from pydantic import ValidationError

from conductor._internal.base import REQUIRED
from conductor._internal.compiled import compile_graph
from conductor._internal.field import Field
from conductor._internal.gate import RefusedError, record_values, validator_of
from conductor._internal.graph import Graph
from conductor._internal.node import (
    FunctionDefinition,
    NodeDefinition,
    SubgraphDefinition,
    declared_inputs,
    read_by_hooks,
)
from conductor._internal.problem import PROBLEMS, InvalidCallError, Problem
from conductor._internal.record import GraphRecord, NodeRecord
from conductor._internal.registry import NodeRegistry
from conductor._internal.run import EndedEarly
from conductor._internal.types import Param, kind_of_value
from conductor.testing import problems

#: One call's inputs, by name.
_Example = Mapping[str, Any]


#: The problems of loading that are a hook's.
_HOOK_CODES = frozenset(kind.code for kind in PROBLEMS if kind.of_a_hook)


class _Cannot:
    """What ``_made_up`` gives for a field no value can be made for."""


_CANNOT = _Cannot()


def check_registry[C](
    registry: NodeRegistry[C],
    /,
    *,
    examples: Mapping[str | tuple[str, int], Sequence[_Example]] = MappingProxyType({}),
    context: C | None = None,
) -> list[Problem]:
    """Every way a node of ``registry`` breaks what a host relies on, all at once; ``[]`` when none does."""
    found: list[Problem] = []
    by_version: dict[tuple[str, int], list[_Example]] = {}
    for key, calls in examples.items():
        if isinstance(key, str):  # the newest version, by the id alone
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
    definition: NodeDefinition, registry: NodeRegistry[Any], examples: Mapping[tuple[str, int], Sequence[_Example]], context: Any,
) -> list[Problem]:
    found = _version_problems(definition, registry, examples.get((definition.id, definition.version), ()), context)
    for older in definition.version_chain[1:]:
        given = examples.get((older.id, older.version), ())
        found += _upgrade_problems(older, registry, given)
        found += _version_problems(older, registry, given, context)
    return found


def _version_problems(version: NodeDefinition, registry: NodeRegistry[Any], given: Sequence[_Example], context: Any) -> list[Problem]:
    """The hooks checked when there is no example, else each example run: what went wrong."""
    if not given:
        return _hook_problems(version, registry)
    return [problem for example in given for problem in _run_problems(version, registry, example, context)]


def _one_node_graph(version: NodeDefinition, constants: Mapping[str, Any]) -> GraphRecord:
    """A graph of this one node, at its version, with ``constants`` typed in, placed under its own id.

    A value with no JSON form can't be typed into a graph: ``InvalidCallError``, ``invalid_value`` at the node and input.
    """
    try:
        written = record_values(version.id, constants)
    except RefusedError as refused:
        raise InvalidCallError(refused.problems) from None
    return GraphRecord(nodes=[NodeRecord(id=version.id, node=version.id, version=version.version, constants=written)])


def _hook_problems(version: NodeDefinition, registry: NodeRegistry[Any]) -> list[Problem]:
    """Load the node with its defaults and a made-up value for each required input its hooks read."""
    match version:
        case FunctionDefinition(hook_parameters=hooks) if hooks:
            pass
        case FunctionDefinition() | SubgraphDefinition():  # no hooks; a subgraph has none of its own
            return []
    read = sorted(read_by_hooks(version))
    values = {name: _made_up(version.inputs[name]) for name in read if version.inputs[name].default is REQUIRED}
    missing = [name for name, value in values.items() if value is _CANNOT]
    if missing:
        return [problems.needs_example(version.id, version.version, inputs=missing)]
    failed = [p for p in Graph.compile(_one_node_graph(version, values), registry).problems if p.type in _HOOK_CODES]
    return [problems.hook_failed(version.id, version.version, reason=_reason([p]), cause=[p.type], problems=[p.model_dump(mode="json")]) for p in failed]


def _upgrade_problems(older: NodeDefinition, registry: NodeRegistry[Any], given: Sequence[_Example]) -> list[Problem]:
    """Place the node at ``older`` with an example's values, or made-up ones, move it to the newest version and load it."""
    if given:
        values = dict(given[0])
    else:
        values = {name: made for name, field in declared_inputs(older).items() if (made := _made_up(field)) is not _CANNOT}
    newest = registry[older.id].version
    try:
        record = _one_node_graph(older, values)
    except InvalidCallError:
        return []  # the example's own run says what can't be typed in
    try:
        upgraded = registry.upgrade(record, older.id)
    except Exception as raised:  # noqa: BLE001  an upgrade's rewrite is the author's code
        return [problems.upgrade_raised(older.id, older.version, newest=newest, error=type(raised).__name__, reason=str(raised))]
    graph_problems = Graph.compile(upgraded, registry).problems
    if not graph_problems:
        return []
    return [problems.upgrade_broken(older.id, older.version, newest=newest, reason=_reason(graph_problems), cause=[p.type for p in graph_problems], problems=[p.model_dump(mode="json") for p in graph_problems])]


def _run_problems(version: NodeDefinition, registry: NodeRegistry[Any], example: _Example, context: Any) -> list[Problem]:
    """Run one example through the engine: what went wrong."""
    match version:
        case SubgraphDefinition():  # a subgraph runs on the nodes of the registry
            definitions, changed = registry.filter(lambda held: held.id != version.id), list[str]()
            definitions.add(version)
        case FunctionDefinition():
            watched, changed = _watched(version)
            definitions = registry.filter(lambda _: False)
            definitions.add(watched)
    try:
        record = _one_node_graph(version, dict(example))
    except InvalidCallError as refused:
        return _not_compiled(version, refused.errors())
    graph = compile_graph(record, definitions, registry.context_type, Graph)
    if graph.problems:
        return _not_compiled(version, graph.problems)
    try:
        run = graph.run(context=context)
    except InvalidCallError as refused:
        call_problems = refused.errors()
        return [problems.example_refused(
            version.id, version.version, reason=_reason(call_problems), cause=[p.type for p in call_problems], problems=[p.model_dump(mode="json") for p in call_problems],
        )]
    found = [
        problems.changed_its_inputs(version.id, version.version, input=name)
        for name in changed
    ]
    match run.state:
        case EndedEarly(status=status, cause=cause):
            found.append(problems.run_failed(version.id, version.version, status=status, reason=_reason([cause]), cause=[cause.type], problems=[cause.model_dump(mode="json")]))
    return found


def _not_compiled(version: NodeDefinition, graph_problems: Sequence[Problem]) -> list[Problem]:
    """The one problem for an example whose graph has problems: ``hook_failed`` when a hook caused every one, else ``run_failed``."""
    reason = _reason(graph_problems)
    cause, written = [p.type for p in graph_problems], [p.model_dump(mode="json") for p in graph_problems]
    if all(p.type in _HOOK_CODES for p in graph_problems):
        return [problems.hook_failed_on_an_example(version.id, version.version, reason=reason, cause=cause, problems=written)]
    return [problems.example_not_compiled(version.id, version.version, reason=reason, cause=cause, problems=written)]


def _reason(behind: Sequence[Problem]) -> str:
    """The ``reason`` of every kind of ``check_registry``: the message of each problem behind it, joined by "; ".

    A node's own exception is ``execution_failed``, whose message does not say which exception it was, so its class
    name is said instead: ``"the node raised ZeroDivisionError"``.
    """
    return "; ".join(f"the node raised {p.ctx['error']}" if p.type == "execution_failed" else p.msg for p in behind)


def _watched(version: FunctionDefinition[..., Any]) -> tuple[FunctionDefinition[..., Any], list[str]]:
    """``version``, called as before, noting each input a call changed; never retried. The run's context, which a
    node that takes it gets first, is passed on and not watched."""
    function = version.function
    changed: list[str] = []

    def watched(*context: Any, **kwargs: Any) -> Any:
        before: dict[str, object] = {}
        for name, value in kwargs.items():
            try:
                kept = copy.deepcopy(value)
            except Exception:  # noqa: BLE001, S112  a value that can't be copied can't be compared
                continue
            if _same(kept, value):  # a value unequal to its own copy (NaN, no __eq__) can't be compared
                before[name] = kept
        try:
            return function(*context, **kwargs)
        finally:
            changed.extend(name for name, value in before.items() if name not in changed and not _same(value, kwargs[name]))

    return dataclasses.replace(version, function=watched, policy=version.policy.model_copy(update={"retries": 0})), changed


def _made_up(field: Field) -> Any:
    """A value of the field within its limits, read by the field's type; ``_CANNOT`` when there is no way to make one."""
    read = validator_of(field.whole).validate_python
    for item in _candidates(field.type.options[0] if len(field.type.options) == 1 else None, field.param):
        try:
            return read([item, item] if field.is_series else item)
        except ValidationError:  # outside a limit, or a pattern it doesn't match: the field says which it takes
            continue
    return _CANNOT


def _candidates(hint: Any, param: Param) -> list[Any]:
    """Values a field of ``hint`` may take, tried in order; the field's own adapter keeps the first it takes."""
    if typing.get_origin(hint) is Literal:
        return [typing.get_args(hint)[0]]
    if not isinstance(hint, type):
        return []
    if issubclass(hint, bool):
        return [True]
    if issubclass(hint, (int, float)):
        low: Any = param.ge if param.ge is not None else param.gt  # a number's limits are numbers: the Param hook refused any other
        high: Any = param.le if param.le is not None else param.lt
        middle = [(low + high) // 2 if issubclass(hint, int) else (low + high) / 2] if low is not None and high is not None else []
        near = [*(() if low is None else (low, low + 1)), *(() if high is None else (high, high - 1))]
        return [*middle, *near, 1, 0, -1]
    if issubclass(hint, str):
        return ["x" * max(1, param.min_length or 1)]
    if issubclass(hint, datetime.datetime):
        return [datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)]
    if issubclass(hint, datetime.date):
        return [datetime.date(2026, 1, 1)]
    return []


def _same(one: Any, other: Any) -> bool:
    if one is other:
        return True
    if isinstance(one, Mapping) and isinstance(other, Mapping):
        return one.keys() == other.keys() and all(_same(one[key], other[key]) for key in one)
    if isinstance(one, list) and isinstance(other, list):
        return len(one) == len(other) and all(_same(a, b) for a, b in zip(one, other, strict=True))
    return kind_of_value(one).untyped_equal(one, other)


__all__ = ["check_registry"]
