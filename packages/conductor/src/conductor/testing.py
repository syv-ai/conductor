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
``example_for_no_node`` without a version and ``waiver_not_needed``, which are ``(node_id,)``:

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
    the body changed a value it was given; the rows of a run can share one.
``run_failed``
    an example didn't run to the end or to a pause.
``pause_untested``
    the body raises ``Ask``, itself or through a function it calls, and no
    example made it pause.
``example_for_no_node``, ``waiver_not_needed``
    an example or a waiver names a node or a failure that isn't there.

``expected_failures`` maps ``"id:code"`` to why it is accepted; it lives in
the test, not on the node, and a waiver for a check that passes is itself a problem.
These codes are ``check_registry``'s, not in ``PROBLEM_CODES``: a developer reads
them in a test run, a host never meets them.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import polars as pl
from pydantic import ValidationError

from conductor._internal.ask import Ask
from conductor._internal.errors import InvalidCallError, Problem
from conductor._internal.fields import FieldSpec, Param, without_none
from conductor._internal.graph import load_graph
from conductor._internal.node_definition import (
    FunctionDefinition,
    GraphDefinition,
    NodeDefinition,
    read_by_hooks,
)
from conductor._internal.record import GraphRecord, NodeRecord, typed_in
from conductor._internal.registry import NodeRegistry

#: One call's inputs, by name.
_Example = Mapping[str, Any]


#: The problems of loading that are a hook's.
_HOOK_CODES = frozenset({"compute_hook_failed", "parameter_name_invalid"})


class _Cannot:
    """What ``_made_up`` gives for a field no value can be made for."""


_CANNOT = _Cannot()


def check_registry(
    registry: NodeRegistry,
    /,
    *,
    examples: Mapping[str | tuple[str, int], Sequence[_Example]] | None = None,
    expected_failures: Mapping[str, str] | None = None,
    from_run: Mapping[type, Any] | None = None,
) -> list[Problem]:
    """Every way a node of ``registry`` breaks what a host relies on, all at once; ``[]`` when none does."""
    found: list[Problem] = []
    by_version: dict[tuple[str, int], list[_Example]] = {}
    for key, calls in (examples or {}).items():
        said = f"{key!r} names no node of the registry; check the id, or remove the example"
        if isinstance(key, str):  # the newest version, by the id alone
            if key not in registry:
                found.append(_problem("example_for_no_node", (key,), said))
                continue
            place = (key, registry[key].version)
        elif key[0] not in registry or key[1] not in registry[key[0]].versions:
            found.append(_problem("example_for_no_node", key, said))
            continue
        else:
            place = key
        by_version.setdefault(place, []).extend(calls)
    for definition in registry:
        found += _problems_of(definition, registry, by_version, from_run or {})
    return _without_waived(found, expected_failures or {})


def _problems_of(
    definition: NodeDefinition, registry: NodeRegistry, examples: Mapping[tuple[str, int], Sequence[_Example]], from_run: Mapping[type, Any],
) -> list[Problem]:
    found, paused = _version_problems(definition, registry, examples.get((definition.id, definition.version), ()), from_run)
    for older in definition.lineage[1:]:
        given = examples.get((older.id, older.version), ())
        found += _upgrade_problems(older, registry, given)
        found += _version_problems(older, registry, given, from_run)[0]
    match definition:
        case FunctionDefinition(function=function):
            if not paused and _may_ask(function):
                found.append(_problem("pause_untested", (definition.id, definition.version), "the body raises Ask, and no example made it pause; add an example that makes it ask"))
        case GraphDefinition():  # its nodes are checked as the registry's own
            pass
    return found


def _version_problems(version: NodeDefinition, registry: NodeRegistry, given: Sequence[_Example], from_run: Mapping[type, Any]) -> tuple[list[Problem], bool]:
    """The hooks checked when there is no example, else each example run: what went wrong, and whether one paused."""
    if not given:
        return _hook_problems(version, registry), False
    found: list[Problem] = []
    paused = False
    for example in given:
        problems, asked = _run_problems(version, registry, example, from_run)
        found += problems
        paused = paused or asked
    return found, paused


def _problem(code: str, loc: tuple[str] | tuple[str, int], message: str, **ctx: Any) -> Problem:
    return Problem(type=code, loc=loc, msg=message, ctx=ctx)


def _placed(version: NodeDefinition, values: Mapping[str, Any]) -> GraphRecord:
    """A graph of this one node, at its version, with ``values`` typed in, placed under its own id.

    A value with no JSON form can't be typed into a graph: ``InvalidCallError``, ``invalid_value`` at the node and input.
    """
    typed, problems = typed_in(version.id, values)
    if problems:
        raise InvalidCallError(problems)
    return GraphRecord(nodes=[NodeRecord(id=version.id, node=version.id, version=version.version, values=typed)])


def _causes(problems: Sequence[Problem]) -> dict[str, Any]:
    return {"cause": [p.type for p in problems], "problems": [p.model_dump(mode="json") for p in problems]}


def _hook_problems(version: NodeDefinition, registry: NodeRegistry) -> list[Problem]:
    """Load the node with its defaults and a made-up value for each required input its hooks read."""
    match version:
        case FunctionDefinition(hook_parameters=hooks) if hooks:
            pass
        case FunctionDefinition() | GraphDefinition():  # no hooks; a stored graph has none of its own
            return []
    read = sorted(read_by_hooks(version))
    values = {name: _made_up(version.inputs[name]) for name in read if version.inputs[name].required}
    missing = [name for name, value in values.items() if value is _CANNOT]
    if missing:
        return [_problem("needs_example", (version.id, version.version), f"its hooks read {missing}, and no value could be made up; give an example", inputs=missing)]
    failed = [p for p in registry.load(_placed(version, values)).problems if p.type in _HOOK_CODES]
    return [_problem("hook_failed", (version.id, version.version), f"a hook failed on the node's defaults and made-up values: {p.msg}", **_causes([p])) for p in failed]


def _upgrade_problems(older: NodeDefinition, registry: NodeRegistry, given: Sequence[_Example]) -> list[Problem]:
    """Place the node at ``older`` with an example's values, or made-up ones, move it to the newest version and load it."""
    if given:
        values = dict(given[0])
    else:
        values = {name: made for name, spec in _inputs_of(older).items() if (made := _made_up(spec)) is not _CANNOT}
    newest = registry[older.id].version
    try:
        placed = _placed(older, values)
    except InvalidCallError:
        return []  # the example's own run says what can't be typed in
    try:
        upgraded = registry.upgrade(placed, older.id)
    except Exception as raised:  # noqa: BLE001  an upgrade's rewrite is the author's code
        return [_problem("upgrade_broken", (older.id, older.version), f"moving to version {newest} raised {type(raised).__name__}: {raised}", error=type(raised).__name__)]
    problems = registry.load(upgraded).problems
    if not problems:
        return []
    said = "; ".join(f"{p.type}: {p.msg}" for p in problems)
    return [_problem("upgrade_broken", (older.id, older.version), f"after moving to version {newest}: {said}", **_causes(problems))]


def _inputs_of(version: NodeDefinition) -> Mapping[str, FieldSpec]:
    """The inputs a version declares; a stored graph's are known only when a graph using it loads."""
    match version:
        case FunctionDefinition():
            return version.inputs
        case GraphDefinition():
            return {}


def _run_problems(version: NodeDefinition, registry: NodeRegistry, example: _Example, from_run: Mapping[type, Any]) -> tuple[list[Problem], bool]:
    """Run one example through the engine: what went wrong, and whether it paused for a person."""
    definitions: dict[str, NodeDefinition]
    match version:
        case GraphDefinition():  # a stored graph runs on the nodes of the registry
            definitions, changed = {d.id: d for d in registry}, list[str]()
            definitions[version.id] = version
        case FunctionDefinition():
            watched, changed = _watched(version)
            definitions = {version.id: watched}
    try:
        placed = _placed(version, dict(example))
    except InvalidCallError as refused:
        return _not_loaded(version, refused.errors()), False
    graph = load_graph(placed, definitions)
    if graph.problems:
        return _not_loaded(version, graph.problems), False
    try:
        run = graph.run(from_run=from_run)
    except InvalidCallError as refused:
        return [_problem("run_failed", (version.id, version.version), "the example was refused: " + "; ".join(p.msg for p in refused.errors()), **_causes(refused.errors()))], False
    found = [
        _problem("changed_its_inputs", (version.id, version.version), f"the body changed {name!r}, a value it was given; rows of a run can share it, so copy the value before changing it", input=name)
        for name in changed
    ]
    if run.status not in ("complete", "waiting"):
        causes = [run.cause] if run.cause is not None else []
        found.append(_problem("run_failed", (version.id, version.version), f"the example ended {run.status}: {run.cause.msg if run.cause else ''}".rstrip(": "), **_causes(causes)))
    # an Ask the run refused was reached all the same; run_failed says what was wrong with it
    return found, run.status == "waiting" or (run.cause is not None and run.cause.type == "invalid_ask")


def _not_loaded(version: NodeDefinition, problems: Sequence[Problem]) -> list[Problem]:
    """The one problem of an example that doesn't load: a hook's when every cause is, else the run's."""
    code = "hook_failed" if all(p.type in _HOOK_CODES for p in problems) else "run_failed"
    return [_problem(code, (version.id, version.version), "the example doesn't load: " + "; ".join(p.msg for p in problems), **_causes(problems))]


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

    return dataclasses.replace(version, function=watched, policy=dataclasses.replace(version.policy, retries=0)), changed


def _made_up(spec: FieldSpec) -> Any:
    """A value of the field within its limits, checked by the field; ``_CANNOT`` when there is no way to make one."""
    if spec.shape == "table":
        return _CANNOT
    for item in _candidates(without_none(spec.type), spec.param):
        try:
            return spec.adapter.validate_python([item, item] if spec.shape == "many" else item)
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


def _may_ask(function: Callable[..., Any]) -> bool:
    """Whether ``function`` names ``Ask``, itself or through a function of its globals or closure it calls.

    A node's function is a plain function or a bound method; any other callable (a ``staticmethod`` object, say) has no
    code to read, so it is taken not to ask.
    """
    if isinstance(function, types.MethodType):
        function = function.__func__
    if not isinstance(function, types.FunctionType):
        return False
    seen: set[types.CodeType] = set()

    def asks(function: types.FunctionType) -> bool:
        if function.__code__ in seen:
            return False
        seen.add(function.__code__)
        names: Mapping[str, Any] = function.__globals__
        helpers = [cell.cell_contents for cell in function.__closure__ or () if _filled(cell)]

        def names_ask(code: types.CodeType) -> bool:
            for name in code.co_names:
                if name == "Ask" or names.get(name) is Ask:
                    return True
                if isinstance(names.get(name), types.FunctionType):
                    helpers.append(names[name])
            return any(names_ask(inner) for inner in code.co_consts if isinstance(inner, types.CodeType))

        return names_ask(function.__code__) or any(isinstance(h, types.FunctionType) and asks(h) for h in helpers)

    return asks(function)


def _filled(cell: types.CellType) -> bool:
    try:
        cell.cell_contents  # noqa: B018
    except ValueError:  # a cell not filled yet
        return False
    return True


def _same(one: Any, other: Any) -> bool:
    if one is other:
        return True
    if isinstance(one, Mapping) and isinstance(other, Mapping):
        return one.keys() == other.keys() and all(_same(one[key], other[key]) for key in one)
    if isinstance(one, pl.DataFrame) and isinstance(other, pl.DataFrame):
        return one.equals(other)
    if isinstance(one, list) and isinstance(other, list):
        return len(one) == len(other) and all(_same(a, b) for a, b in zip(one, other, strict=True))
    try:
        return bool(one == other)
    except (ValueError, TypeError):
        return one is other


def _without_waived(found: list[Problem], waived: Mapping[str, str]) -> list[Problem]:
    # a code holds no ':', and a node id may, so the id is all before the last one
    by_place = {(node_id, code): key for key in waived for node_id, _, code in [key.rpartition(":")]}
    used = {place for problem in found if (place := (problem.loc[0], problem.type)) in by_place}
    kept = [problem for problem in found if (problem.loc[0], problem.type) not in by_place]
    for place in sorted(by_place.keys() - used):
        key = by_place[place]
        kept.append(_problem("waiver_not_needed", (place[0],), f"{key} no longer fails; remove the waiver ({waived[key]})", waiver=key))
    return kept


__all__ = ["check_registry"]
