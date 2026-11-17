"""Whether an edge may join two types.

When a graph is compiled, each edge is checked: can a value of the output's type be passed to the input's
type? An ``int`` output may go to a ``float`` input, but a ``str`` output may not go to an ``int`` input.
Pydantic checks values, not pairs of types, so this check is written here.

A type with a limit, choices or another check, at any depth, follows one rule. A value of the right class may go
to it, and each value is checked on its call. A ``Literal`` source names its values, so one more rule holds for it,
whatever the target: it lands when the target's own check takes one of its values and their classes fit the
target. So ``Literal["x"]`` doesn't land on ``Literal["ida", "bo"]``, and ``Literal[1]`` doesn't land on
``Literal[True]``, though pydantic takes ``1`` as ``True``.
"""

from __future__ import annotations

import typing
from typing import Any, Literal, TypeVar, get_args, get_origin

from .hint import a_number, aliased, is_alias, literal_classes, union_members
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
    (bare, param, others), source = split_hint(target), split_hint(source)[0]
    checked = bool(param.limits or param.choices is not None or others)
    # the values a Literal source names, or a union of them, maybe with None; a str beside them may be any text
    members = union_members(source)
    names_its_values = any(get_origin(member) is Literal for member in members) and all(
        member is type(None) or get_origin(member) is Literal for member in members
    )
    if names_its_values and not isinstance(bare, TypeVar) and (bare is not Any or checked):
        # A source that names its values lands when the target's own check, the one the run gives each value
        # (Type.takes_value), takes one of them, and their classes fit the target: Literal[1] lands on float, but
        # not on Literal[True], which pydantic takes as equal. It needs no check on its call when the target's check
        # takes each value as it is, of the same class, since no other value can come.
        from .type import Type  # Type is built on fit, so it is imported here

        values = [value for member in members for value in (get_args(member) or (None,))]
        checks, classes = Type(target), literal_classes(source)
        if not any(checks.takes_value(value) for value in values):
            return False
        if not fit(target, classes, bound, False):
            return False
        return not exact or all(
            checks.takes_value(value, as_it_is=True) for value in values if value is not None
        )
    if checked:
        # a limit, a choice or a validator may refuse a value of the right class, so it is checked on each call
        return not exact and fit(bare, source, bound, False)
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
        # a value of the right class may be one of them, so it is checked on each call
        return not exact and fit(literal_classes(target), source, bound, False)
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
