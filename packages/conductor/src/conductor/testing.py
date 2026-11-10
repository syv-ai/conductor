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
import inspect
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import polars as pl
from pydantic import ValidationError

from conductor._internal.ask import Ask
from conductor._internal.errors import InvalidCallError, Problem
from conductor._internal.fields import FieldSpec, without_none
from conductor._internal.graph import load_graph
from conductor._internal.node_definition import NodeDefinition
from conductor._internal.record import GraphRecord, NodeRecord
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
    examples = examples or {}
    found: list[Problem] = []
    for key in examples:
        node_id, version = (key, None) if isinstance(key, str) else key
        if node_id not in registry or (version is not None and version not in registry[node_id].versions):
            found.append(_problem("example_for_no_node", node_id, version, f"{key!r} names no node of the registry; check the id, or remove the example"))
    for definition in registry:
        found += _problems_of(definition, registry, examples, from_run or {})
    return _without_waived(found, expected_failures or {})


def _problems_of(
    definition: NodeDefinition[..., Any], registry: NodeRegistry, examples: Mapping[Any, Sequence[_Example]], from_run: Mapping[type, Any],
) -> list[Problem]:
    found: list[Problem] = []
    paused = False
    version: NodeDefinition[..., Any] | None = definition
    while version is not None:
        given = [*(examples.get(definition.id, ()) if version is definition else ()), *examples.get((definition.id, version.version), ())]
        if not given:
            found += _hook_problems(version, registry)
        if version is not definition:
            found += _upgrade_problems(version, registry, given)
        for example in given:
            problems, asked = _run_problems(version, registry, example, from_run)
            found += problems
            paused = paused or (asked and version is definition)
        version = version.replaces
    if definition.function is not None and not paused and _may_ask(definition.function):
        found.append(_problem("pause_untested", definition.id, definition.version, "the body raises Ask, and no example made it pause; add an example that makes it ask"))
    return found


def _problem(code: str, node_id: str, version: int | None, message: str, **ctx: Any) -> Problem:
    return Problem(type=code, loc=(node_id,) if version is None else (node_id, version), msg=message, ctx=ctx)


def _placed(version: NodeDefinition[..., Any], values: Mapping[str, Any]) -> GraphRecord:
    """A graph of this one node, at its version, with ``values`` typed in, placed under its own id."""
    return GraphRecord(nodes=[NodeRecord(id=version.id, node=version.id, version=version.version, values=values)])


def _causes(problems: Sequence[Problem]) -> dict[str, Any]:
    return {"cause": [p.type for p in problems], "problems": [p.model_dump(mode="json") for p in problems]}


def _hook_problems(version: NodeDefinition[..., Any], registry: NodeRegistry) -> list[Problem]:
    """Load the node with its defaults and a made-up value for each required input its hooks read."""
    hooks = [hook for hook in (version.compute_added_inputs, version.compute_outputs) if hook is not None]
    if not hooks:
        return []
    read = sorted({name for hook in hooks for name, p in inspect.signature(hook).parameters.items() if p.annotation not in (type, "type")})
    values = {name: _made_up(version.inputs[name]) for name in read if version.inputs[name].required}
    missing = [name for name, value in values.items() if value is _CANNOT]
    if missing:
        return [_problem("needs_example", version.id, version.version, f"its hooks read {missing}, and no value could be made up; give an example", inputs=missing)]
    failed = [p for p in registry.load(_placed(version, values)).problems if p.type in _HOOK_CODES]
    return [_problem("hook_failed", version.id, version.version, f"a hook failed on the node's defaults and made-up values: {p.msg}", **_causes([p])) for p in failed]


def _upgrade_problems(older: NodeDefinition[..., Any], registry: NodeRegistry, given: Sequence[_Example]) -> list[Problem]:
    """Place the node at ``older`` with an example's values, or made-up ones, move it to the newest version and load it."""
    if given:
        values = dict(given[0])
    else:
        values = {name: made for name, spec in older.inputs.items() if (made := _made_up(spec)) is not _CANNOT}
    newest = registry[older.id].version
    try:
        upgraded = registry.upgrade(_placed(older, values), older.id)
    except Exception as raised:  # noqa: BLE001  an upgrade's rewrite is the author's code
        return [_problem("upgrade_broken", older.id, older.version, f"moving to version {newest} raised {type(raised).__name__}: {raised}", error=type(raised).__name__)]
    problems = registry.load(upgraded).problems
    if not problems:
        return []
    said = "; ".join(f"{p.type}: {p.msg}" for p in problems)
    return [_problem("upgrade_broken", older.id, older.version, f"after moving to version {newest}: {said}", **_causes(problems))]


def _run_problems(version: NodeDefinition[..., Any], registry: NodeRegistry, example: _Example, from_run: Mapping[type, Any]) -> tuple[list[Problem], bool]:
    """Run one example through the engine: what went wrong, and whether it paused for a person."""
    values = dict(example)
    if version.function is None:  # a stored graph runs on the nodes of the registry
        definitions, changed = {d.id: d for d in registry}, list[str]()
        definitions[version.id] = version
    else:
        watched, changed = _watched(version)
        definitions = {version.id: watched}
    graph = load_graph(_placed(version, values), definitions)
    if graph.problems:
        code = "hook_failed" if all(p.type in _HOOK_CODES for p in graph.problems) else "run_failed"
        return [_problem(code, version.id, version.version, "the example doesn't load: " + "; ".join(p.msg for p in graph.problems), **_causes(graph.problems))], False
    try:
        run = graph.run(from_run=from_run)
    except InvalidCallError as refused:
        return [_problem("run_failed", version.id, version.version, "the example was refused: " + "; ".join(p.msg for p in refused.errors()), **_causes(refused.errors()))], False
    found = [
        _problem("changed_its_inputs", version.id, version.version, f"the body changed {name!r}, a value it was given; rows of a run can share it, so copy the value before changing it", input=name)
        for name in changed
    ]
    if run.status not in ("complete", "waiting"):
        causes = [run.cause] if run.cause is not None else []
        found.append(_problem("run_failed", version.id, version.version, f"the example ended {run.status}: {run.cause.msg if run.cause else ''}".rstrip(": "), **_causes(causes)))
    # an Ask the run refused was reached all the same; run_failed says what was wrong with it
    return found, run.status == "waiting" or (run.cause is not None and run.cause.type == "invalid_ask")


def _watched(version: NodeDefinition[..., Any]) -> tuple[NodeDefinition[..., Any], list[str]]:
    """``version`` calling its function once per unit as before, noting each input the call changed; never retried."""
    function = typing.cast(Callable[..., Any], version.function)
    changed: list[str] = []

    def watched(**kwargs: Any) -> Any:
        before = {}
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
    item = _one(without_none(spec.type), spec.param.constraints)
    if item is _CANNOT:
        return _CANNOT
    try:
        return spec.adapter.validate_python([item, item] if spec.shape == "many" else item)
    except ValidationError:  # a pattern, say, that a made-up value doesn't match
        return _CANNOT


def _one(hint: Any, limits: Mapping[str, Any]) -> Any:
    if typing.get_origin(hint) is Literal:
        return typing.get_args(hint)[0]
    if not isinstance(hint, type):
        return _CANNOT
    if issubclass(hint, bool):
        return True
    if issubclass(hint, (int, float)):
        low = limits.get("ge", limits["gt"] + 1 if "gt" in limits else None)
        return low if low is not None else min(1, limits.get("le", 1), limits.get("lt", 2) - 1)
    if issubclass(hint, str):
        return "x" * max(1, limits.get("min_length", 1))
    if issubclass(hint, datetime.datetime):
        return datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    if issubclass(hint, datetime.date):
        return datetime.date(2026, 1, 1)
    return _CANNOT


def _may_ask(function: Callable[..., Any], seen: set[types.CodeType] | None = None) -> bool:
    """Whether ``function`` names ``Ask``, itself or through a function of its globals or closure it calls."""
    seen = set() if seen is None else seen
    code = getattr(function, "__code__", None)
    if not isinstance(code, types.CodeType) or code in seen:
        return False
    seen.add(code)
    names: Mapping[str, Any] = getattr(function, "__globals__", {})
    helpers = [cell.cell_contents for cell in getattr(function, "__closure__", None) or () if _filled(cell)]

    def names_ask(code: types.CodeType) -> bool:
        for name in code.co_names:
            if name == "Ask" or names.get(name) is Ask:
                return True
            if isinstance(names.get(name), types.FunctionType):
                helpers.append(names[name])
        return any(names_ask(inner) for inner in code.co_consts if isinstance(inner, types.CodeType))

    return names_ask(code) or any(isinstance(h, types.FunctionType) and _may_ask(h, seen) for h in helpers)


def _filled(cell: Any) -> bool:
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
    used = {key for problem in found if (key := f"{problem.loc[0]}:{problem.type}") in waived}
    kept = [problem for problem in found if f"{problem.loc[0]}:{problem.type}" not in waived]
    for key in sorted(waived.keys() - used):
        node_id = key.partition(":")[0]
        kept.append(_problem("waiver_not_needed", node_id, None, f"{key} no longer fails; remove the waiver ({waived[key]})", waiver=key))
    return kept


__all__ = ["check_registry"]
