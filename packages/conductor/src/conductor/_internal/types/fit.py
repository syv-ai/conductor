"""Edge fit: can a value of one type be passed to an input of another type? An ``int`` fits ``float``; a ``str``
does not fit ``int``. Pydantic has no such check, so it is written here.
"""

from __future__ import annotations

import typing
from typing import Any, Literal, TypeVar, get_args, get_origin

from conductor._internal.types.hint import a_number, aliased, is_alias, union_members
from conductor._internal.types.host_type import accepted_classes
from conductor._internal.types.param import split_hint


def fit(target: Any, source: Any, bound: dict[TypeVar, Any], exact: bool) -> bool:
    """Says whether a value of ``source`` may land on ``target``, binding ``target``'s type variables as it walks.

    Args:
        target: The hint asked for, which may hold type variables.
        source: The hint arriving, which holds none.
        bound: What each variable is bound to so far, added to here; two places sharing a variable join.
        exact: Ask instead whether every ``source`` value already is a ``target`` as it is, with nothing to read.
    """
    target, source = split_hint(target)[0], split_hint(source)[0]
    if target == source:
        return True
    if is_alias(target) or is_alias(source):  # an alias is read as what it stands for
        return fit(aliased(target) if is_alias(target) else target, aliased(source) if is_alias(source) else source, bound, exact)
    if isinstance(target, TypeVar):
        if target.__bound__ is not None and not fit(target.__bound__, source, {}, exact):
            return False
        bound[target] = bound[target] | source if target in bound else source
        return True
    if target is Any:
        return True
    if source is Any:
        return not exact
    targets, sources = union_members(target), [option for option in union_members(source) if option is not type(None)]
    if len(targets) > 1 or len(sources) != len(union_members(source)):
        for one in sources:
            for option in targets:
                if fit(option, one, trial := dict(bound), exact):
                    bound.update(trial)
                    break
            else:
                return False
        return True
    if get_origin(source) is Literal:
        if get_origin(target) is Literal:  # a narrower Literal lands on one holding every value it may be
            return {(type(v), v) for v in get_args(source)} <= {(type(v), v) for v in get_args(target)}
        return all(fit(target, type(value), bound, exact) for value in get_args(source))
    target_class, source_class = get_origin(target) or target, get_origin(source) or source
    if typing.is_typeddict(source):
        source_fields = typing.get_type_hints(source)
        if typing.is_typeddict(target):  # read by its keys: every key the target names, each of a type that fits
            expected = typing.get_type_hints(target)
            same = not exact or expected.keys() == source_fields.keys()  # a read drops the keys the target doesn't name
            return same and target.__required_keys__ <= source.__required_keys__ and all(key in source_fields and fit(hint, source_fields[key], bound, exact) for key, hint in expected.items())
        if target_class is not dict:
            return False
        key, value = get_args(target) or (Any, Any)
        return fit(key, str, bound, exact) and all(fit(value, hint, bound, exact) for hint in source_fields.values())
    if typing.is_typeddict(target) or not (isinstance(target_class, type) and isinstance(source_class, type)):
        return False
    if issubclass(source_class, bool) and a_number(target_class):
        return False
    widened = not exact and ((target_class is float and issubclass(source_class, int)) or issubclass(source_class, accepted_classes(target_class)))
    if not (issubclass(source_class, target_class) or widened):
        return False
    wanted_args, held_args = get_args(target), get_args(source)
    if not held_args:  # a bare generic arriving says nothing of its items
        return not (exact and wanted_args)
    return len(wanted_args) != len(held_args) or all(fit(a, b, bound, exact) for a, b in zip(wanted_args, held_args, strict=True))
