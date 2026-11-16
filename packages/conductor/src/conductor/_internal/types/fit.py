"""Whether an edge may join two types.

When a graph is compiled, each edge is checked: can a value of the output's type be passed to the input's
type? An ``int`` output may go to a ``float`` input, but a ``str`` output may not go to an ``int`` input.
Pydantic checks values, not pairs of types, so this check is written here.

A type with a limit, choices or another check, at any depth, follows one rule. A value of the right class may go
to it, and each value is checked on its call. A ``Literal`` source names its values, so it may not go to a type
whose own check takes none of them.
"""

from __future__ import annotations

import typing
from typing import Any, Literal, TypeVar, get_args, get_origin

from .hint import a_number, aliased, is_alias, literal_classes, union_members, union_of
from .host_type import accepted_classes
from .param import split_hint


def fit(target: Any, source: Any, bound: dict[TypeVar, Any], exact: bool) -> bool:
    """Says whether a value of type ``source`` may be passed where ``target`` is expected.

    Along the way it binds the type variables in ``target``. For ``target`` ``list[T]`` and ``source``
    ``list[int]``, it returns ``True`` and binds ``T`` to ``int``.

    Args:
        target: The type expected, which may have type variables.
        source: The type arriving, which has none.
        bound: What each type variable is bound to so far. New bindings are added to it. When a variable appears
            in two places, it is bound to the union of what arrives at each.
        exact: Ask a stricter question instead: is every ``source`` value already a ``target`` value as it is, so
            that it needs no checking or converting?
    """
    if target == source:
        return True
    (bare, param, others), source = split_hint(target), _as_one_literal(split_hint(source)[0])
    if param.limits or param.choices is not None or others:
        # a limit, a choice or a validator may refuse a value of the right class, so it is checked on each call,
        # and an edge whose values it refuses all is refused here
        return not exact and _may_be_taken(target, source) and fit(bare, source, bound, False)
    target = bare
    if target == source:
        return True
    if is_alias(target) or is_alias(source):  # an alias stands for another type, so compare that
        return fit(
            aliased(target) if is_alias(target) else target,
            aliased(source) if is_alias(source) else source,
            bound,
            exact,
        )
    if isinstance(target, TypeVar):
        if target.__bound__ is not None and not fit(target.__bound__, source, {}, exact):
            return False
        bound[target] = bound[target] | source if target in bound else source
        return True
    if target is Any:
        return True
    if source is Any:
        return not exact
    targets, sources = (
        union_members(target),
        [option for option in union_members(source) if option is not type(None)],
    )
    if len(targets) > 1 or len(union_members(source)) > 1:
        for one in sources:
            for option in targets:
                if fit(option, one, trial := dict(bound), exact):
                    bound.update(trial)
                    break
            else:
                return False
        return True
    if get_origin(target) is Literal:
        if not _may_be_taken(target, source):
            return False
        if get_origin(source) is Literal:
            # exact when every value is one of the target's as it is, of the same class: True is not 1
            given_values, target_values = get_args(source), get_args(target)
            return not exact or {(type(v), v) for v in given_values} <= {
                (type(v), v) for v in target_values
            }
        # a value of the right class may be one of them, so it is checked on each call
        return not exact and fit(literal_classes(target), source, bound, False)
    if get_origin(source) is Literal:
        return all(fit(target, type(value), bound, exact) for value in get_args(source))
    target_class, source_class = get_origin(target) or target, get_origin(source) or source
    if typing.is_typeddict(source):
        source_fields = typing.get_type_hints(source, include_extras=True)
        # a record fits by its keys: every key the target names must be there, with a type that fits
        if typing.is_typeddict(target):
            expected = typing.get_type_hints(target, include_extras=True)
            # a check drops the keys the target doesn't name, so the value isn't the same as it was
            same = not exact or expected.keys() == source_fields.keys()
            return (
                same
                and target.__required_keys__ <= source.__required_keys__
                and all(
                    key in source_fields and fit(hint, source_fields[key], bound, exact)
                    for key, hint in expected.items()
                )
            )
        if target_class is not dict:
            return False
        key, value = get_args(target) or (Any, Any)
        return fit(key, str, bound, exact) and all(
            fit(value, hint, bound, exact) for hint in source_fields.values()
        )
    if typing.is_typeddict(target) or not (
        isinstance(target_class, type) and isinstance(source_class, type)
    ):
        return False
    if issubclass(source_class, bool) and a_number(target_class):
        return False
    widened = not exact and (
        (target_class is float and issubclass(source_class, int))
        or issubclass(source_class, accepted_classes(target_class))
    )
    if not (issubclass(source_class, target_class) or widened):
        return False
    expected_args, given_args = get_args(target), get_args(source)
    if not given_args:  # a bare generic like ``list`` says nothing about its items
        return not (exact and expected_args)
    return len(expected_args) != len(given_args) or all(
        fit(a, b, bound, exact) for a, b in zip(expected_args, given_args, strict=True)
    )


def _as_one_literal(source: Any) -> Any:
    """Returns ``source`` with its ``Literal``s joined into one, and a ``None`` among their values as ``| None``.

    ``Literal["x"] | Literal["ida"]`` gives ``Literal["x", "ida"]``, and ``Literal["x", None]`` gives
    ``Literal["x"] | None``. Then a source that names its values is one ``Literal`` wherever ``fit`` looks.
    """
    members = union_members(source)
    values = [
        value for member in members if get_origin(member) is Literal for value in get_args(member)
    ]
    if not values:
        return source
    others = [member for member in members if get_origin(member) is not Literal]
    if None in values:
        others = [*dict.fromkeys([*others, type(None)])]
    kept = tuple(value for value in values if value is not None)
    return union_of([*([Literal[kept]] if kept else []), *others])


def _may_be_taken(target: Any, source: Any) -> bool:
    """Says whether a value of ``source`` may be taken by ``target``'s own check.

    Only a source of ``Literal`` values, with ``None`` maybe, names its values, so only it can be refused here: when
    the target takes none of them, no call could ever get a value. Each value is checked as the run checks the
    value an edge gives (``Type.takes_value``), so the edge and the run follow one rule. ``Literal["x"]`` into
    ``Literal["ida", "bo"]`` is refused, while ``str``, ``Literal["ida", "x"]``, and ``Literal[1]`` into
    ``Annotated[float, Param(choices=[1.0])]`` may be taken.
    """
    # Type is built on fit, so it is imported here
    from .type import Type

    members = union_members(source)
    if any(member is not type(None) and get_origin(member) is not Literal for member in members):
        return True
    values = [value for member in members for value in (get_args(member) or (None,))]
    checked = Type(target)
    return any(checked.takes_value(value) for value in values)
