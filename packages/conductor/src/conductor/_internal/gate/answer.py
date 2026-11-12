"""The answer door: a person's answer to a call that waits, read into the fields it fills.

Example: ``approve(line: str) -> bool`` raises ``Ask("Approve b?")``. ``run.answer(waiting_id, True)`` gives the
door ``True``, and it returns ``{"result": True}``, the outputs the call then has. For a node with an ``Asked``
parameter, ``ok: Annotated[bool, Asked(approval)]``, the field is the parameter: the door returns ``{"ok": True}``,
and the node is then called with ``ok=True``. ``"yes"`` is refused with
``invalid_answer``, "Input should be a valid boolean": an answer is read strictly from its JSON form, so a yes or
no is ``true`` or ``false``, never a text that spells one.

A node with several outputs is answered with a mapping by output name, one question per output. A question may be
left out when it has a default (its field's, or one the ``Ask`` set), is one of an exclusive group, or takes
``None`` and has no choices; of an exclusive group exactly one is answered. Choices, the type's or the ``Ask``'s,
limit what an answer may be.

An answer is read from JSON by the validator that reads what a node returns (``rules.output_validator_of``): the
answer is the value of the function that asked, as a return would be.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError
from pydantic_core import PydanticCustomError, PydanticSerializationError, to_json

from conductor._internal.ask import QuestionOverrides, answer_required
from conductor._internal.base import SKIPPED, Loc, comma_separated
from conductor._internal.field import Field, FieldDescription
from conductor._internal.gate.refused import RefusedError
from conductor._internal.gate.rules import output_validator_of
from conductor._internal.problem import catalogue as problem
from conductor._internal.types import Type, untyped_as_json


class AnswerDoor:
    """Reads a person's answer to a waiting call of one node into the fields it fills: the node's outputs, or its
    ``Asked`` parameter. Built once per node.

    Raises (when called):
        RefusedError: With an ``invalid_answer`` for every fault, quoting nothing of the answer: the answer is not
            a mapping where there are several questions, names a question there is none of, leaves out one that
            must be answered, gives a value its type refuses or that is not one of the choices, or answers none
            or two of an exclusive group.
    """

    def __init__(self, fields: Mapping[str, Field], questions: Mapping[str, FieldDescription]) -> None:
        """Builds the door for a node.

        Args:
            fields: The fields the answer fills, by name.
            questions: The question about each field as its type sets it (``field_questions``), by name.
        """
        self._fields = fields
        self._questions = questions

    def __call__(self, answer: Any, overrides: Mapping[str, QuestionOverrides], *, loc: Loc, waiting_id: str) -> dict[str, Any]:
        """Returns the fields the answer fills, by name, read from ``answer``.

        Args:
            answer: The answer: a value for one question, a mapping by question name for several.
            overrides: What the ``Ask`` set on each question, by field name.
            loc: Where the call that waits is, for the problems.
            waiting_id: The waiting id the answer was given for, for the problems.

        Raises:
            RefusedError: See the class.
        """
        read, faults = self._read(answer, overrides)
        if faults:
            raise RefusedError(problem.invalid_answer(*loc, waiting_id=waiting_id, reason=reason, question=question or None) for question, reason in faults)
        return read

    def _read(self, answer: Any, overrides: Mapping[str, QuestionOverrides]) -> tuple[dict[str, Any], list[tuple[str | None, str]]]:
        """The fields read from ``answer``, and every fault as ``(question, reason)``, the question ``None`` for a
        fault of the whole answer."""
        outputs = self._fields
        if len(outputs) == 1:
            ((name, field),) = outputs.items()
            try:
                return {name: answer_value(field, answer, overrides.get(name, {}).get("choices", self._questions[name].choices))}, []
            except PydanticCustomError as refused:
                return {}, [(None, refused.message())]
        if answer is None and not outputs:
            return {}, []  # a node that returns nothing is answered with nothing: the person only says to go on
        if not isinstance(answer, Mapping):
            return {}, [(None, f"answer each question by name: a mapping of {comma_separated(outputs, empty='nothing')}")]
        faults: list[tuple[str | None, str]] = [(str(name), f"there is no question '{name}'") for name in answer if name not in outputs]
        read: dict[str, Any] = {}
        groups: dict[str, list[str]] = {}
        for name, field in outputs.items():
            question, override, group = self._questions[name], overrides.get(name, {}), field.param.exclusive_group
            choices = override.get("choices", question.choices)
            try:
                if name in answer:
                    read[name] = answer_value(field, answer[name], choices)
                elif group is not None:
                    read[name] = SKIPPED
                elif field.default is SKIPPED and "default" not in override:
                    read[name] = SKIPPED  # left out, a field whose default is SKIPPED is a branch not taken
                elif not answer_required(field, override, limited=choices is not None):
                    read[name] = from_json(field.whole, override.get("default", question.default))  # its default, else None for a type that takes it
                else:
                    faults.append((name, f"'{name}' was not answered"))
            except PydanticCustomError as refused:
                faults.append((name, f"'{name}': {refused.message()}"))
            if group is not None:
                groups.setdefault(group, []).append(name)
        for members in groups.values():
            if sum(name in answer for name in members) != 1:
                faults.append((None, f"answer exactly one of {', '.join(members)}"))
        return read, faults


def answer_value(field: Field, value: Any, choices: list[dict[str, Any]] | None) -> Any:
    """Reads one question's answer into the form a run keeps, held to ``choices``.

    ``answer_value(field, "bo", [{"id": "ida", ...}, {"id": "bo", ...}])`` for a field ``str`` is ``"bo"``;
    ``"al"`` is not one of the choices. A ``Series`` output picks several, each once.

    Args:
        field: The field the question is about.
        value: The answer to it.
        choices: The choices it is limited to, each its record with the id in its JSON form; ``None`` for none.

    Raises:
        PydanticCustomError: The answer is not a value of the field's type, not one of the choices, or, for a
            pick of several, picks a choice twice. Its message says which, quoting nothing of the answer.
    """
    read = from_json(field.whole, value)
    if choices is not None:
        picked: Any = field.whole.to_json(read)
        picks = picked if field.is_series else [picked]
        ids = [choice["id"] for choice in choices]
        if any(item not in ids for item in picks):
            raise PydanticCustomError("not_a_choice", "the answer is not one of the choices")
        if len({repr(item) for item in picks}) != len(picks):
            raise PydanticCustomError("picked_twice", "a choice is picked twice")
    return read


def from_json(value_type: Type, value: Any) -> Any:
    """Reads ``value`` strictly from its JSON form by ``value_type``, into the form a run keeps.

    ``from_json(Type(bool), True)`` is ``True``; ``from_json(Type(bool), "yes")`` is refused. A date is read from
    the ISO text JSON writes it as, and a ``Series``' list is kept as a run stores a returned one.

    Raises:
        PydanticCustomError: The value has no JSON form (``no_json_form``), or the type refuses it
            (``not_its_type``); the message is the type's reason, quoting nothing of the value.
    """
    try:
        written = to_json(value, inf_nan_mode="constants", fallback=untyped_as_json)
    except PydanticSerializationError:
        raise PydanticCustomError("no_json_form", "the value has no JSON form") from None
    try:
        return output_validator_of(value_type).validate_json(written, strict=True)
    except ValidationError as error:
        raise PydanticCustomError("not_its_type", value_type.refusal(error.errors(include_url=False, include_input=False)).reason) from None

