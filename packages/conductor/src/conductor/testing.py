"""``check_registry``: one test that every node of a registry keeps what a host relies on.

A node package or a host puts one line in its test suite::

    from conductor.testing import check_registry

    def test_nodes():
        assert check_registry(registry, examples={"upper": [{"text": "hi"}]}) == []

It checks what ``@node`` can't see in a declaration. A body is called only
for a node given an example, since a node may send mail; ``examples`` maps a
node id to the inputs of its calls, by name, and ``(id, version)`` to those
of an older version. Each example runs through the engine as a graph of that
one node, its inputs typed in, given ``from_run={T: value}`` as ``graph.run``
is.

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
import types
import typing
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from conductor._internal import kit_problem
from conductor._internal.checked_graph import load_graph
from conductor._internal.field import Field
from conductor._internal.graph import Graph
from conductor._internal.graph_record import GraphRecord, NodeRecord, values_as_json
from conductor._internal.node_definition import (
    FunctionDefinition,
    NodeDefinition,
    StoredGraphDefinition,
    declared_inputs,
    read_by_hooks,
)
from conductor._internal.problem import PROBLEMS, InvalidCallError, Problem
from conductor._internal.registry import NodeRegistry
from conductor._internal.run_status import EndedEarly
from conductor._internal.sentinels import REQUIRED
from conductor._internal.value_type import FieldValueError, Param, kind_of_value

#: One call's inputs, by name.
_Example = Mapping[str, Any]


#: The problems of loading that are a hook's.
_HOOK_CODES = frozenset(kind.code for kind in PROBLEMS if kind.of_a_hook)


class _Cannot:
    """What ``_made_up`` gives for a field no value can be made for."""


_CANNOT = _Cannot()


def check_registry(
    registry: NodeRegistry,
    /,
    *,
    examples: Mapping[str | tuple[str, int], Sequence[_Example]] = types.MappingProxyType({}),
    from_run: Mapping[type, Any] = types.MappingProxyType({}),
) -> list[Problem]:
    """Every way a node of ``registry`` breaks what a host relies on, all at once; ``[]`` when none does."""
    found: list[Problem] = []
    by_version: dict[tuple[str, int], list[_Example]] = {}
    for key, calls in examples.items():
        if isinstance(key, str):  # the newest version, by the id alone
            if key not in registry:
                found.append(kit_problem.example_for_no_node(key, named=repr(key)))
                continue
            place = (key, registry[key].version)
        elif key[0] not in registry or key[1] not in registry[key[0]].versions:
            found.append(kit_problem.example_for_no_node(*key, named=repr(key)))
            continue
        else:
            place = key
        by_version.setdefault(place, []).extend(calls)
    for definition in registry:
        found += _problems_of(definition, registry, by_version, from_run)
    return found


def _problems_of(
    definition: NodeDefinition, registry: NodeRegistry, examples: Mapping[tuple[str, int], Sequence[_Example]], from_run: Mapping[type, Any],
) -> list[Problem]:
    found = _version_problems(definition, registry, examples.get((definition.id, definition.version), ()), from_run)
    for older in definition.lineage[1:]:
        given = examples.get((older.id, older.version), ())
        found += _upgrade_problems(older, registry, given)
        found += _version_problems(older, registry, given, from_run)
    return found


def _version_problems(version: NodeDefinition, registry: NodeRegistry, given: Sequence[_Example], from_run: Mapping[type, Any]) -> list[Problem]:
    """The hooks checked when there is no example, else each example run: what went wrong."""
    if not given:
        return _hook_problems(version, registry)
    return [problem for example in given for problem in _run_problems(version, registry, example, from_run)]


def _one_node_graph(version: NodeDefinition, values: Mapping[str, Any]) -> GraphRecord:
    """A graph of this one node, at its version, with ``values`` typed in, placed under its own id.

    A value with no JSON form can't be typed into a graph: ``InvalidCallError``, ``invalid_value`` at the node and input.
    """
    typed, problems = values_as_json(version.id, values)
    if problems:
        raise InvalidCallError(problems)
    return GraphRecord(nodes=[NodeRecord(id=version.id, node=version.id, version=version.version, values=typed)])


def _hook_problems(version: NodeDefinition, registry: NodeRegistry) -> list[Problem]:
    """Load the node with its defaults and a made-up value for each required input its hooks read."""
    match version:
        case FunctionDefinition(hook_parameters=hooks) if hooks:
            pass
        case FunctionDefinition() | StoredGraphDefinition():  # no hooks; a stored graph has none of its own
            return []
    read = sorted(read_by_hooks(version))
    values = {name: _made_up(version.inputs[name]) for name in read if version.inputs[name].default is REQUIRED}
    missing = [name for name, value in values.items() if value is _CANNOT]
    if missing:
        return [kit_problem.needs_example(version.id, version.version, inputs=missing)]
    failed = [p for p in registry.load(_one_node_graph(version, values)).problems if p.type in _HOOK_CODES]
    return [kit_problem.hook_failed(version.id, version.version, reason=p.msg, cause=[p.type], problems=[p.model_dump(mode="json")]) for p in failed]


def _upgrade_problems(older: NodeDefinition, registry: NodeRegistry, given: Sequence[_Example]) -> list[Problem]:
    """Place the node at ``older`` with an example's values, or made-up ones, move it to the newest version and load it."""
    if given:
        values = dict(given[0])
    else:
        values = {name: made for name, field in declared_inputs(older).items() if (made := _made_up(field)) is not _CANNOT}
    newest = registry[older.id].version
    try:
        placed = _one_node_graph(older, values)
    except InvalidCallError:
        return []  # the example's own run says what can't be typed in
    try:
        upgraded = registry.upgrade(placed, older.id)
    except Exception as raised:  # noqa: BLE001  an upgrade's rewrite is the author's code
        return [kit_problem.upgrade_raised(older.id, older.version, newest=newest, error=type(raised).__name__, reason=str(raised))]
    problems = registry.load(upgraded).problems
    if not problems:
        return []
    said = "; ".join(f"{p.type}: {p.msg}" for p in problems)
    return [kit_problem.upgrade_broken(older.id, older.version, newest=newest, reason=said, cause=[p.type for p in problems], problems=[p.model_dump(mode="json") for p in problems])]


def _run_problems(version: NodeDefinition, registry: NodeRegistry, example: _Example, from_run: Mapping[type, Any]) -> list[Problem]:
    """Run one example through the engine: what went wrong."""
    definitions: dict[str, NodeDefinition]
    match version:
        case StoredGraphDefinition():  # a stored graph runs on the nodes of the registry
            definitions, changed = {d.id: d for d in registry}, list[str]()
            definitions[version.id] = version
        case FunctionDefinition():
            watched, changed = _watched(version)
            definitions = {version.id: watched}
    try:
        placed = _one_node_graph(version, dict(example))
    except InvalidCallError as refused:
        return _not_loaded(version, refused.errors())
    graph = load_graph(placed, definitions, Graph)
    if graph.problems:
        return _not_loaded(version, graph.problems)
    try:
        run = graph.run(from_run=from_run)
    except InvalidCallError as refused:
        faults = refused.errors()
        return [kit_problem.example_refused(
            version.id, version.version, reason="; ".join(p.msg for p in faults), cause=[p.type for p in faults], problems=[p.model_dump(mode="json") for p in faults],
        )]
    found = [
        kit_problem.changed_its_inputs(version.id, version.version, input=name)
        for name in changed
    ]
    match run.state:
        case EndedEarly(status=status, cause=cause):
            # a body's own exception reaches the problem as its class only; the message says which
            said = f"the node raised {cause.ctx['error']}" if cause.type == "execution_failed" else cause.msg
            found.append(kit_problem.run_failed(version.id, version.version, status=status, reason=said, cause=[cause.type], problems=[cause.model_dump(mode="json")]))
    return found


def _not_loaded(version: NodeDefinition, problems: Sequence[Problem]) -> list[Problem]:
    """The one problem of an example that doesn't load: a hook's when every cause is, else the run's."""
    said = "; ".join(p.msg for p in problems)
    cause, written = [p.type for p in problems], [p.model_dump(mode="json") for p in problems]
    if all(p.type in _HOOK_CODES for p in problems):
        return [kit_problem.hook_failed_on_an_example(version.id, version.version, reason=said, cause=cause, problems=written)]
    return [kit_problem.example_not_loaded(version.id, version.version, reason=said, cause=cause, problems=written)]


def _watched(version: FunctionDefinition[..., Any]) -> tuple[FunctionDefinition[..., Any], list[str]]:
    """``version``, called once per unit as before, noting each input the call changed; never retried."""
    function = version.function
    changed: list[str] = []

    def watched(**kwargs: Any) -> Any:
        before: dict[str, object] = {}
        for name, value in kwargs.items():
            if name not in version.from_run:
                try:
                    kept = copy.deepcopy(value)
                except Exception:  # noqa: BLE001, S112  a value that can't be copied can't be compared
                    continue
                if _same(kept, value):  # a value unequal to its own copy (NaN, no __eq__) can't be compared
                    before[name] = kept
        try:
            return function(**kwargs)
        finally:
            changed.extend(name for name, value in before.items() if name not in changed and not _same(value, kwargs[name]))

    return dataclasses.replace(version, function=watched, policy=version.policy.model_copy(update={"retries": 0})), changed


def _made_up(field: Field) -> Any:
    """A value of the field within its limits, checked by the field; ``_CANNOT`` when there is no way to make one."""
    for item in _candidates(field.type.options[0] if len(field.type.options) == 1 else None, field.param):
        try:
            return field.whole.read([item, item] if field.many else item)
        except FieldValueError:  # outside a limit, or a pattern it doesn't match: the field says which it takes
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
