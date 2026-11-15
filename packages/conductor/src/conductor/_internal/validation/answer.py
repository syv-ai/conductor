"""The answer validator: a person's answer to a call that waits, read into the node's outputs.

Example: ``approve(line: str) -> bool`` raises ``Ask("Approve b?")``. ``run.answer(waiting_id, True)`` gives the
validator ``True``, and it returns ``{"result": True}``, the outputs the call then has. ``"yes"`` is refused with
``invalid_answer``, "Input should be a valid boolean": an answer is read strictly, as its JSON text would be, so a
yes or no is ``true`` or ``false``, never a text that spells one.

A node with several outputs is answered with a mapping by output name, one question per output. A question left
out gives what the question says (``as_asked``): its default, a branch not chosen when it is not ``required``, else
it is a fault; of an exclusive group exactly one is answered. Choices, the type's or the ``Ask``'s,
limit what an answer may be.

An answer is read as its JSON text would be (``validate_as_json``), by the validator that reads what a node returns
(``rules.schema_output_validator_of``): the answer is the value of the function that asked, as a return would be.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import MISSING, JsonValue, ValidationError
from pydantic_core import PydanticCustomError, PydanticSerializationError

from conductor._internal.ask import as_asked
from conductor._internal.base import SKIPPED, Loc, validate_as_json
from conductor._internal.field import Field
from conductor._internal.problem import InvalidCallError, call_problems
from conductor._internal.types import (
    Choice,
    Type,
    check_one_of,
    choice_texts,
    pick_texts,
    untyped_as_json,
)
from conductor._internal.utils import comma_separated

from .rules import schema_output_validator_of


class AnswerValidator:
    """Reads a person's answer to a waiting call of one node into the node's outputs. Built once per node.

    Raises (when called):
        InvalidCallError: With an ``invalid_answer`` for every fault, quoting nothing of the answer: the answer is not
            a mapping where there are several questions, names a question there is none of, leaves out one that
            must be answered, gives a value its type refuses or that is not one of the choices, or answers none
            or two of an exclusive group.
    """

    def __init__(self, outputs: Mapping[str, Field]) -> None:
        """Builds the validator for a node.

        Args:
            outputs: The node's outputs, which the answer fills, by name.
        """
        self._outputs = outputs
        groups: dict[str, list[str]] = {}
        for name, field in outputs.items():
            if field.param.exclusive_group is not None:
                groups.setdefault(field.param.exclusive_group, []).append(name)
        self._groups = groups  # the outputs of each exclusive group, by group

    def __call__(
        self,
        answer: Any,
        choices: Mapping[str, tuple[Choice, ...]],
        defaults: Mapping[str, JsonValue],
        *,
        loc: Loc,
        waiting_id: str,
    ) -> dict[str, Any]:
        """Returns the outputs the answer fills, by name, read from ``answer``.

        Args:
            answer: The answer: a value for one question, a mapping by question name for several.
            choices: The choices the call's ``Ask`` offered, by output name.
            defaults: The answer the call's ``Ask`` proposed, by output name.
            loc: Where the call that waits is, for the problems.
            waiting_id: The waiting id the answer was given for, for the problems.

        Raises:
            InvalidCallError: See the class.
        """
        read, faults = self._read(answer, choices, defaults)
        if faults:
            raise InvalidCallError(
                call_problems.invalid_answer(
                    *loc, waiting_id=waiting_id, reason=reason, question=question or None
                )
                for question, reason in faults
            )
        return read

    def _read(
        self,
        answer: Any,
        choices: Mapping[str, tuple[Choice, ...]],
        defaults: Mapping[str, JsonValue],
    ) -> tuple[dict[str, Any], list[tuple[str | None, str]]]:
        """The outputs read from ``answer``, and every fault as ``(question, reason)``, the question ``None`` for a
        fault of the whole answer. A question left out gives what it says: its default, else a branch not chosen
        when it is not ``required``, else nothing, which is a fault."""
        outputs = self._outputs
        if len(outputs) == 1:
            ((name, field),) = outputs.items()
            try:
                return {name: answer_value(field, answer, choices.get(name))}, []
            except PydanticCustomError as refused:
                return {}, [(None, refused.message())]
        if answer is None and not outputs:
            return (
                {},
                [],
            )  # a node that returns nothing is answered with nothing: the person only says to go on
        if not isinstance(answer, Mapping):
            return {}, [
                (
                    None,
                    f"answer each question by name: a mapping of {comma_separated(outputs, empty='nothing')}",
                )
            ]
        faults: list[tuple[str | None, str]] = [
            (str(name), f"there is no question '{name}'") for name in answer if name not in outputs
        ]
        read: dict[str, Any] = {}
        for name, field in outputs.items():
            try:
                if name in answer:
                    read[name] = answer_value(field, answer[name], choices.get(name))
                    continue
                question = as_asked(field, choices, defaults)
                if question.default is not MISSING:
                    read[name] = read_as_answer(field.whole, question.default)
                elif not question.required:
                    read[name] = SKIPPED
                else:
                    faults.append((name, f"'{name}' was not answered"))
            except PydanticCustomError as refused:
                faults.append((name, f"'{name}': {refused.message()}"))
        for members in self._groups.values():
            if sum(name in answer for name in members) != 1:
                faults.append((None, f"answer exactly one of {', '.join(members)}"))
        return read, faults


def answer_value(field: Field, value: Any, choices: tuple[Choice, ...] | None) -> Any:
    """Reads one question's answer into the form a run keeps, held to its choices.

    ``answer_value(field, "bo", (Choice("ida", "Ida B."), Choice("bo", "Bo K.")))`` for a field ``str`` is ``"bo"``;
    ``"al"`` is not one of the choices. The validator that reads the answer checks it against the type's own
    choices; ``check_picks`` checks the rest, the ``Ask``'s choices and a ``Series``' picks each once, and the value
    read is written for it only when there is one of the two to check.

    Args:
        field: The field the question is about.
        value: The answer to it.
        choices: The choices the ``Ask`` offers for it, each id in its JSON form; ``None`` when it offers none.

    Raises:
        PydanticCustomError: The answer is not a value of the field's type, not one of the choices, or, for a
            pick of several, picks a choice twice. Its message says which, quoting nothing of the answer.
    """
    read = read_as_answer(field.whole, value)
    if _has_picks_to_check(field, choices):
        check_picks(field, field.whole.serializer.to_python(read, mode="json"), choices)
    return read


def check_picks(field: Field, written: Any, choices: tuple[Choice, ...] | None) -> None:
    """Checks ``written``, an answer the field's type read, in its JSON form, for what the type's validator does
    not check: that it is one of the ``Ask``'s ``choices``, when it offers some, and that a ``Series`` picks each
    choice once. ``None`` is no pick.

    A pick is compared by its text (``pick_texts``), the rule a ``Param``'s choices hold an input or a return to,
    on the value read: ``1`` answers a ``float`` choice ``1.0``. Only an answer picks each once: a node may return
    a ``Series`` that holds a value twice.

    Raises:
        PydanticCustomError: A pick is not one of the ``Ask``'s choices (``not_a_choice``), else one is picked
            twice (``picked_twice``); the message quotes nothing of the answer.
    """
    if not _has_picks_to_check(field, choices):
        return
    picks = pick_texts(written, several=field.is_series)
    if choices is not None:
        check_one_of(choice_texts(choices), picks)
    if len(set(picks)) != len(picks):
        raise PydanticCustomError("picked_twice", "a choice is picked twice")


def _has_picks_to_check(field: Field, choices: tuple[Choice, ...] | None) -> bool:
    """Whether ``check_picks`` has anything to check an answer for: the ``Ask``'s choices, or a ``Series`` whose
    items have choices."""
    return choices is not None or (field.is_series and field.type.choices is not None)


def read_as_answer(value_type: Type, value: Any) -> Any:
    """Reads ``value`` as a person's answer of ``value_type`` is read, into the form a run keeps.

    ``read_as_answer(Type(bool), True)`` is ``True``; ``read_as_answer(Type(bool), "yes")`` is refused. The value
    is read as its JSON text would be (``validate_as_json``), so an answer a host gives from Python and the same
    answer sent over HTTP read the same: a ``date`` is read from its ISO text too, and a ``Series``' list is kept
    as a run stores a returned one. An ``Ask``'s default and its choices are read by
    the same validator.

    Raises:
        PydanticCustomError: The value has no JSON form (``no_json_form``), or the type refuses it
            (``not_its_type``); the message is the type's reason, quoting nothing of the value.
    """
    try:
        return validate_as_json(
            schema_output_validator_of(value_type), value, fallback=untyped_as_json
        )
    except PydanticSerializationError:
        raise PydanticCustomError("no_json_form", "the value has no JSON form") from None
    except PydanticCustomError as refused:  # a frame whose column would read back changed
        raise PydanticCustomError("not_its_type", refused.message()) from None
    except ValidationError as error:
        raise PydanticCustomError("not_its_type", value_type.refusal(error).reason) from None
