"""The validators of asking a person: an ``Ask`` a node raises, its choices and default checked when it is raised,
and a person's answer to a call that waits, read into the node's outputs.

The ``Ask``. Example: ``assign(ticket: str) -> str`` raises ``Ask("Who takes it?", choices=[Choice("ida", "Ida B."),
"bo"])``. The validator returns the choices and defaults the call keeps by output name, the choices for ``result`` the
two as the output's type reads them (``read_choices`` with the pydantic validator of the output), each id in its JSON
form. Given ``choices=[1, 2]`` it raises ``InvalidCallError`` with ``invalid_ask``: a choice is what an answer picks,
so each is read strictly from its JSON form, and ``1`` is not a ``str``. A default is held to every rule an answer
is, the ``Ask``'s own choices included.

A ``NodeError``'s ``ctx`` is the other value a node raises; it is read where it is made, by ``NodeError`` itself, so
a ``ctx`` that is not JSON fails at the node's own ``raise``.

The answer. Example: ``approve(line: str) -> bool`` raises ``Ask("Approve b?")``. ``run.answer(waiting_id, True)``
gives the validator ``True``, and it returns ``{"result": True}``, the outputs the call then has. ``"yes"`` is refused
with ``invalid_answer``, "Input should be a valid boolean": an answer is read strictly, as its JSON text would be, so a
yes or no is ``true`` or ``false``, never a text that spells one.

A node with several outputs is answered with a mapping by output name, one question per output. A question left
out gives what the question says (``as_asked``): its default, a branch not chosen when it is not ``required``, else
it is a fault; of an exclusive group exactly one is answered. Choices, the type's or the ``Ask``'s, limit what an
answer may be.

An answer is read as its JSON text would be (``validate_as_json``), by the validator that reads what a node returns
(``schema_output_validator_of``): the answer is the value of the function that asked, as a return would be.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import MISSING, JsonValue, ValidationError
from pydantic_core import PydanticCustomError, PydanticSerializationError

from conductor._internal.base import REQUIRED, SKIPPED, Loc, validate_as_json
from conductor._internal.field import Field, exclusive_groups, groups_without_one_value
from conductor._internal.problem import InvalidCallError, call_problems, run_problems
from conductor._internal.types import (
    Choice,
    Type,
    check_one_of,
    choice_texts,
    pick_texts,
    read_choices,
    untyped_as_json,
)
from conductor._internal.utils import comma_separated
from conductor._internal.validation import schema_output_validator_of

from .question import as_asked
from .type import Ask


class AskValidator:
    """Checks an ``Ask`` raised for a node against the node's outputs, which its answer fills, and reads its choices
    and default by output name. Built once per node.

    Raises (when called):
        InvalidCallError: The ``Ask``'s choices or default don't fit the outputs (``invalid_ask``, whose ``reason``
            says which part, quoting no value).
    """

    def __init__(self, outputs: Mapping[str, Field]) -> None:
        """Builds the validator for a node with ``outputs``, by name."""
        self._outputs = outputs

    def __call__(self, ask: Ask) -> tuple[dict[str, tuple[Choice, ...]], dict[str, JsonValue]]:
        """Returns what the ``Ask`` set on each output's question, which the call keeps until it is answered.

        Args:
            ask: What the node raised.

        Returns:
            The choices the ``Ask`` offered, by output name, each id read by the output's type in its JSON form; and
            the answer it proposed, by output name, in its JSON form. An output it set nothing on is not named.

        Raises:
            InvalidCallError: See the class.
        """
        outputs = self._outputs
        choices: dict[str, tuple[Choice, ...]] = {}
        defaults: dict[str, JsonValue] = {}
        if ask.choices is not None:
            for name, given in self._by_output(ask.choices, "choices").items():
                value_type = outputs[
                    name
                ].type  # a Series' choices are its items', so each id is read as one item
                try:
                    choices[name] = read_choices(
                        given, schema_output_validator_of(value_type), value_type.to_json
                    )
                except PydanticCustomError as refused:
                    raise InvalidCallError(
                        [
                            run_problems.invalid_ask(
                                reason=f"the choices for '{name}' are refused: {refused.message()}"
                            )
                        ]
                    ) from None
        if ask.default is not REQUIRED:
            for name, default in self._by_output(ask.default, "default").items():
                # a default would give the output a value whenever another of its group is answered
                if outputs[name].param.exclusive_group is not None:
                    reason = f"'{name}' is one of an exclusive group, so an Ask proposes no default for it: an answer that left it out would give it the default, and no other output of the group could be answered"
                    raise InvalidCallError([run_problems.invalid_ask(reason=reason)])
                defaults[name] = _proposed(outputs[name], default, choices.get(name))
        return choices, defaults

    def _by_output(self, given: Any, argument: str) -> dict[str, Any]:
        """One ``Ask`` argument by output name: a lone output's as itself, several outputs' from the mapping given.

        Raises:
            InvalidCallError: The argument doesn't have the answer's shape, or names an output the node doesn't have
                (``invalid_ask``).
        """
        outputs = self._outputs
        if len(outputs) == 1:
            if argument == "choices" and isinstance(given, Mapping):
                raise InvalidCallError(
                    [
                        run_problems.invalid_ask(
                            reason="the node asks one question, so it is answered with a value, and choices= is a list"
                        )
                    ]
                )
            return {next(iter(outputs)): given}
        if not isinstance(given, Mapping):
            reason = f"the node has {len(outputs)} outputs, so it is answered by output name, and {argument}= goes by output name too"
            raise InvalidCallError([run_problems.invalid_ask(reason=reason)])
        for name in given:
            if name not in outputs:
                reason = f"the {argument} names '{name}', which is not an output; the outputs are {comma_separated(outputs)}"
                raise InvalidCallError([run_problems.invalid_ask(reason=reason)])
        return dict(given)


def undeclared_ask(ask: Ask) -> tuple[dict[str, tuple[Choice, ...]], dict[str, JsonValue]]:
    """The ``Ask`` validator of a node that does not say it asks (no ``asks=True``): it refuses
    every ``Ask``.

    Raises:
        InvalidCallError: Always, with ``ask_not_declared``.
    """
    raise InvalidCallError([run_problems.ask_not_declared()])


def _proposed(field: Field, default: Any, choices: tuple[Choice, ...] | None) -> JsonValue:
    """The default an ``Ask`` proposes for one output, in JSON form, held to the rules an answer is.

    Args:
        field: The output.
        default: The proposed answer as given.
        choices: The choices the same ``Ask`` offers for the output, if any.

    Raises:
        InvalidCallError: An answer could not be the default (``invalid_ask``).
    """
    try:
        written: JsonValue = field.whole.to_json(_read_as_answer(field.whole, default))
        _check_picks(field, written, choices)
    except PydanticCustomError as refused:
        raise InvalidCallError(
            [
                run_problems.invalid_ask(
                    reason=f"the default for '{field.name}' is not an answer it takes: {refused.message()}"
                )
            ]
        ) from None
    return written


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
        self._groups = exclusive_groups(outputs)

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
                return {name: _answer_value(field, answer, choices.get(name))}, []
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
                    read[name] = _answer_value(field, answer[name], choices.get(name))
                    continue
                question = as_asked(field, choices, defaults)
                if question.default is not MISSING:
                    read[name] = _read_as_answer(field.whole, question.default)
                elif not question.required:
                    read[name] = SKIPPED
                else:
                    faults.append((name, f"'{name}' was not answered"))
            except PydanticCustomError as refused:
                faults.append((name, f"'{name}': {refused.message()}"))
        for fault in groups_without_one_value(self._groups, answer):
            faults.append((None, f"answer exactly one of {comma_separated(fault.members)}"))
        return read, faults


def _answer_value(field: Field, value: Any, choices: tuple[Choice, ...] | None) -> Any:
    """Reads one question's answer into the form a run keeps, held to its choices.

    ``_answer_value(field, "bo", (Choice("ida", "Ida B."), Choice("bo", "Bo K.")))`` for a field ``str`` is ``"bo"``;
    ``"al"`` is not one of the choices. The validator that reads the answer checks it against the type's own
    choices; ``_check_picks`` checks the rest, the ``Ask``'s choices and a ``Series``' picks each once, and the value
    read is written for it only when there is one of the two to check.

    Args:
        field: The field the question is about.
        value: The answer to it.
        choices: The choices the ``Ask`` offers for it, each id in its JSON form; ``None`` when it offers none.

    Raises:
        PydanticCustomError: The answer is not a value of the field's type, not one of the choices, or, for a
            pick of several, picks a choice twice. Its message says which, quoting nothing of the answer.
    """
    read = _read_as_answer(field.whole, value)
    if _has_picks_to_check(field, choices):
        _check_picks(field, field.whole.serializer.to_python(read, mode="json"), choices)
    return read


def _check_picks(field: Field, written: Any, choices: tuple[Choice, ...] | None) -> None:
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
    """Whether ``_check_picks`` has anything to check an answer for: the ``Ask``'s choices, or a ``Series`` whose
    items have choices."""
    return choices is not None or (field.is_series and field.type.choices is not None)


def _read_as_answer(value_type: Type, value: Any) -> Any:
    """Reads ``value`` as a person's answer of ``value_type`` is read, into the form a run keeps.

    ``_read_as_answer(Type(bool), True)`` is ``True``; ``_read_as_answer(Type(bool), "yes")`` is refused. The value
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
