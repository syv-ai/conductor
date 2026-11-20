"""Whether an edge may join two types.

When a graph is compiled, each edge is checked: can a value of the output's type be passed to the input's
type? An ``int`` output may go to a ``float`` input, but a ``str`` output may not go to an ``int`` input.
Pydantic checks values, not pairs of types, so this check is written here.

A type with a limit, choices or another check, at any depth, follows one rule. A value of the right class may go
to it, and each value is checked on its call.

One more rule holds where a side names its values: a ``Literal`` source, or a target with choices. The edge lands
only on a value both sides take. A ``Literal`` source lands when the target's own check takes one of its values and
their classes fit the target. So ``Literal["x"]`` doesn't land on ``Literal["ida", "bo"]``, and ``Literal[1]``
doesn't land on ``Literal[True]``, though pydantic takes ``1`` as ``True``. Any other source lands on a target with
choices when its own check takes one of them. So ``int`` doesn't land on ``int | Annotated[str,
Param(choices=["a"])]``, whose one value is text, but does on ``Annotated[float, Param(choices=[1, 2.5])]``.
"""

from __future__ import annotations

from typing import Any, Literal, TypeVar, get_args, get_origin

from .hint import (
    a_number,
    aliased,
    field_hints,
    has_own_config,
    is_alias,
    is_typed_dict,
    literal_classes,
    revalidates_instances,
    types_and_left_out,
    union_members,
    union_of,
)
from .host_type import accepted_classes
from .param import split_hint


def fit(target: Any, source: Any, bound: dict[TypeVar, Any], exact: bool) -> bool:
    """Says whether a value of type ``source`` may be passed where ``target`` is expected.

    Along the way it binds the type variables in ``target``. For ``target`` ``list[T]`` and ``source``
    ``list[int]``, it returns ``True`` and binds ``T`` to ``int``.

    A ``None`` that arrives as the whole value never reaches an input that doesn't take ``None``: the node isn't
    called for it, and a loop's ``None`` items are left out of the list an input collects. So ``source`` is
    compared without that ``None``, and ``int | None`` fits ``int``. Inside a value, ``None`` is a value like any
    other, which ``target`` must take: ``list[int | None]`` doesn't fit ``list[int]``. The one place the whole
    value's ``None`` still counts is a ``Literal`` source, which lands when the target takes one of its values:
    ``Literal["x"] | None`` lands on a target that takes ``None`` beside its choices.

    Args:
        target: The type expected, which may have type variables.
        source: The type arriving, which has none.
        bound: What each type variable is bound to so far. New bindings are added to it. When a variable appears
            in two places, it is bound to the union of what arrives at each.
        exact: Ask a stricter question instead: is every ``source`` value already a ``target`` value as it is, so
            that it needs no checking or converting? A ``TypedDict`` read by a pydantic config of its own holds
            no other type, since the config may change a value, as ``str_to_lower`` does; so does a model or
            pydantic dataclass that revalidates instances (``revalidate_instances``).
    """
    if target == source:
        return True
    whole, may_be_none = _split_off_none(source)
    return _fit(target, whole, bound, exact, may_be_none=may_be_none)


def _split_off_none(hint: Any) -> tuple[Any, bool]:
    """Returns ``hint`` without the ``None`` its whole value may be, and whether it may be ``None``.

    ``int | None`` gives ``int``, and ``Literal["a", None]`` gives ``Literal["a"]``, both with ``True``.
    ``list[int | None]`` is given back as it is, with ``False``, and ``None`` itself as it is, with ``True``.
    """
    members, may_be_none = [], False
    for member in union_members(split_hint(hint)[0]):
        if is_alias(member):
            member, alias_may_be_none = _split_off_none(aliased(member))
            may_be_none |= alias_may_be_none
        if get_origin(member) is Literal:
            values = tuple(value for value in get_args(member) if value is not None)
            may_be_none |= len(values) < len(get_args(member))
            member = Literal[values] if values else type(None)
        if member is type(None):
            may_be_none = True
        else:
            members.append(member)
    return (union_of(members) if members else hint), may_be_none


def _fit(
    target: Any, source: Any, bound: dict[TypeVar, Any], exact: bool, *, may_be_none: bool = False
) -> bool:
    """Says what ``fit`` says, for a ``source`` without the ``None`` its whole value may be.

    ``may_be_none`` says whether the whole value may be ``None``, which only a ``Literal`` source asks.
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
    # a target with a bare type variable among its members binds what arrives, member by member, below
    binds = any(isinstance(member, TypeVar) for member in union_members(bare))
    if names_its_values and not binds and (bare is not Any or checked):
        # A source that names its values lands when the target's own check, the one the run gives each value
        # (Type.takes_value), takes one of them, and their classes fit the target: Literal[1] lands on float, but
        # not on Literal[True], which pydantic takes as equal. It needs no check on its call when the target's check
        # takes each value as it is, of the same class, since no other value can come.
        from .type import Type  # Type is built on fit, so it is imported here

        values = [value for member in members for value in (get_args(member) or (None,))]
        checks, classes = Type(target), literal_classes(source)
        landing = [*values, None] if may_be_none else values
        if not any(checks.takes_value(value) for value in landing):
            return False
        if not _fit(target, classes, bound, False):
            return False
        return not exact or all(checks.takes_value(value, as_it_is=True) for value in values)
    if checked:
        # A limit, a choice or a validator may refuse a value of the right class, so it is checked on each call. A
        # target's choices name its values, so a source lands only when its own check takes one of them.
        from .type import Type  # Type is built on fit, so it is imported here

        if param.choices is not None and not any(
            Type(source).takes_value(choice.id) for choice in Type(target).choices or ()
        ):
            return False
        return not exact and _fit(bare, source, bound, False)
    target = bare
    if target == source:
        return True
    if is_alias(target) or is_alias(source):  # an alias stands for another type, so compare that
        return _fit(
            aliased(target) if is_alias(target) else target,
            aliased(source) if is_alias(source) else source,
            bound,
            exact,
        )
    if isinstance(target, TypeVar):
        if target.__bound__ is not None and not _fit(target.__bound__, source, {}, exact):
            return False
        bound[target] = bound[target] | source if target in bound else source
        return True
    if target is Any:
        return True
    if source is Any:
        return not exact
    targets, sources = union_members(target), union_members(source)
    if len(targets) > 1 or len(sources) > 1:
        # Each arriving member goes to a member that holds it as it is, else to a bare type variable, else to any
        # member that may take it. So a type variable binds only what no other member holds: fed int | None,
        # T | None binds T to int, and fed int, T | float binds T to int, since float takes an int only by converting.
        variables = [option for option in targets if isinstance(option, TypeVar)]
        plain = [option for option in targets if not isinstance(option, TypeVar)]
        tries = [(option, True) for option in plain]
        tries += [(option, exact) for option in (*variables, *plain)]
        for one in sources:
            for option, holding in tries:
                if _fit(option, one, trial := dict(bound), holding):
                    bound.update(trial)
                    break
            else:
                return False
        return True
    if get_origin(target) is Literal:
        # a value of the right class may be one of them, so it is checked on each call
        return not exact and _fit(literal_classes(target), source, bound, False)
    target_class, source_class = get_origin(target) or target, get_origin(source) or source
    if is_typed_dict(source):
        given = types_and_left_out(field_hints(source))
        # a TypedDict fits by its keys: every key the target requires must be there for sure, and every key both
        # name must have a type that fits; a key the target may do without may be missing
        if is_typed_dict(target):
            if exact and has_own_config(target_class):
                # a row read by a config of its own, such as str_to_lower, may change a value of another type
                return False
            expected = types_and_left_out(field_hints(target))
            # a check drops the keys the target doesn't name, so the value isn't the same as it was
            same = not exact or expected.keys() == given.keys()
            return same and all(
                left_out
                if key not in given
                else (left_out or not given[key][1]) and _fit(hint, given[key][0], bound, exact)
                for key, (hint, left_out) in expected.items()
            )
        if target_class is not dict:
            return False
        key, value = get_args(target) or (Any, Any)
        return _fit(key, str, bound, exact) and all(
            _fit(value, hint, bound, exact) for hint, _ in given.values()
        )
    if is_typed_dict(target) or not (
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
    if exact and source_class is not target_class and revalidates_instances(target_class):
        # pydantic reads a subclass's instance again by the target's own config, which may change it
        return False
    expected_args, given_args = get_args(target), get_args(source)
    if not given_args:  # a bare generic like ``list`` says nothing about its items
        return not (exact and expected_args)
    return len(expected_args) != len(given_args) or all(
        _fit(a, b, bound, exact) for a, b in zip(expected_args, given_args, strict=True)
    )
