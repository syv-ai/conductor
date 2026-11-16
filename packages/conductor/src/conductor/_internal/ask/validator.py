"""The two checks on asking a person: one on the ``Ask`` a node raises, one on the answer a person gives.

``AskValidator`` runs when a node raises ``Ask``. It makes sure the choices and the default fit the node's outputs.
Say ``assign(ticket: str) -> str`` raises ``Ask("Who?", choices=[1, 2])``. The answer has to be a ``str``, so no
one could ever pick ``1`` or ``2``. The ``Ask`` is refused with ``invalid_ask``.

``AnswerValidator`` runs when a person answers. It turns the answer into the node's outputs. Say
``approve(line: str) -> bool`` is waiting. The answer ``True`` gives the outputs ``{"result": True}``. The answer
``"yes"`` is refused with ``invalid_answer``, since a yes or no is ``True`` or ``False``, not a word that means one.

An answer is read strictly, the way its JSON text would be read. That way an answer given from Python and the same
answer sent over HTTP are read the same. It is read by the same rules as a value the node returns, since the
answer stands in for that value.
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
from conductor._internal.validation import Validator, schema_output_validator_of

from .question import as_asked
from .type import Ask


class AskValidator(Validator[Ask, tuple[dict[str, tuple[Choice, ...]], dict[str, JsonValue]]]):
    """A validator for the ``Ask`` a node raises.

    The answer to an ``Ask`` fills the node's outputs, so the ``Ask``'s choices and default must fit them. They
    must have the right shape, name only real outputs, and hold values each output's type accepts. The validator
    also turns them into the form the waiting call keeps.

    One validator is built for each node. If the node doesn't say ``asks=True``, it refuses every ``Ask``.

    Raises (when called):
        InvalidCallError: ``ask_not_declared`` if the node doesn't say it asks. ``invalid_ask`` if the choices or
            the default don't fit the outputs. Its ``reason`` says which part is wrong, without quoting the value.
    """

    def __init__(self, outputs: Mapping[str, Field], *, asks: bool) -> None:
        """Builds the validator for one node.

        Args:
            outputs: The node's outputs, by name.
            asks: Whether the node says it may ask, with ``asks=True``.
        """
        self._outputs = outputs
        self._asks = asks

    def __call__(self, ask: Ask) -> tuple[dict[str, tuple[Choice, ...]], dict[str, JsonValue]]:
        """Checks ``ask`` and returns what it sets on each question.

        Args:
            ask: The ``Ask`` the node raised.

        Returns:
            Two mappings by output name. The first holds the choices the ``Ask`` offers, each id read by the
            output's type and kept in its JSON form. The second holds the default it proposes, in its JSON form. An
            output the ``Ask`` sets nothing for is left out of both.

        Raises:
            InvalidCallError: See the class.
        """
        if not self._asks:
            raise InvalidCallError([run_problems.ask_not_declared()])
        outputs = self._outputs
        choices: dict[str, tuple[Choice, ...]] = {}
        defaults: dict[str, JsonValue] = {}
        if ask.choices is not None:
            for name, given in self._by_output(ask.choices, "choices").items():
                # On a Series, the choices are for its items, so each id is read as one item.
                value_type = outputs[name].type
                try:
                    choices[name] = read_choices(
                        given, schema_output_validator_of(value_type), value_type.to_json
                    )
                except PydanticCustomError as refused:
                    raise InvalidCallError(
                        [
                            run_problems.invalid_ask(
                                reason=f"the choices for '{name}' are wrong ({refused.message()})"
                            )
                        ]
                    ) from None
        if ask.default is not REQUIRED:
            for name, default in self._by_output(ask.default, "default").items():
                # A default would give this output a value even when the person picks another of its group.
                if outputs[name].param.exclusive_group is not None:
                    reason = f"'{name}' is in an exclusive group, and an output of a group can't have a default"
                    raise InvalidCallError([run_problems.invalid_ask(reason=reason)])
                defaults[name] = _proposed(outputs[name], default, choices.get(name))
        return choices, defaults

    def _by_output(self, given: Any, argument: str) -> dict[str, Any]:
        """Returns one ``Ask`` argument, ``choices`` or ``default``, by output name.

        For a node with one output, the argument is that output's value. For a node with several, the argument must
        already be a mapping by output name.

        Raises:
            InvalidCallError: ``invalid_ask`` if the argument has the wrong shape for the node, or names an output
                the node doesn't have.
        """
        outputs = self._outputs
        if len(outputs) == 1:
            if argument == "choices" and isinstance(given, Mapping):
                raise InvalidCallError(
                    [
                        run_problems.invalid_ask(
                            reason="the node has one output, so choices= is a list, not a mapping"
                        )
                    ]
                )
            return {next(iter(outputs)): given}
        if not isinstance(given, Mapping):
            reason = f"the node has {len(outputs)} outputs, so {argument}= must be a mapping by output name"
            raise InvalidCallError([run_problems.invalid_ask(reason=reason)])
        for name in given:
            if name not in outputs:
                reason = f"{argument}= names '{name}', which isn't an output. The outputs are {comma_separated(outputs)}"
                raise InvalidCallError([run_problems.invalid_ask(reason=reason)])
        return dict(given)


def _proposed(field: Field, default: Any, choices: tuple[Choice, ...] | None) -> JsonValue:
    """Checks the default an ``Ask`` proposes for one output, and returns it as JSON.

    A default is a ready-made answer, so it must be an answer a person could give. It must be a value of the
    output's type, and one of the ``Ask``'s choices if it offers any.

    Args:
        field: The output the default is for.
        default: The default, as the ``Ask`` gave it.
        choices: The choices the same ``Ask`` offers for this output, or ``None``.

    Returns:
        The default in its JSON form.

    Raises:
        InvalidCallError: ``invalid_ask`` if a person couldn't give this default as an answer.
    """
    try:
        written: JsonValue = field.whole.to_json(_read_as_answer(field.whole, default))
        _check_picks(field, written, choices)
    except PydanticCustomError as refused:
        raise InvalidCallError(
            [
                run_problems.invalid_ask(
                    reason=f"the default for '{field.name}' isn't an answer a person could give ({refused.message()})"
                )
            ]
        ) from None
    return written


class AnswerValidator(Validator[Any, dict[str, Any]]):
    """A validator for a person's answer to a node that raised ``Ask``.

    It checks the answer and turns it into the node's outputs. A node with one output is answered with a plain
    value. A node with several is answered with a mapping by output name, one entry per question. An entry may be
    left out when its question has a default, or when the question is a branch the person may leave unchosen.

    One validator is built for each answer, from the questions the call asked.

    Raises (when called):
        InvalidCallError: One ``invalid_answer`` problem per fault, none of them quoting the answer. The faults
            are: no mapping where there are several questions, a name that isn't a question, a required question
            left out, a value its type or its choices refuse, and not exactly one answer in an exclusive group.
    """

    def __init__(
        self,
        outputs: Mapping[str, Field],
        choices: Mapping[str, tuple[Choice, ...]],
        defaults: Mapping[str, JsonValue],
        *,
        loc: Loc,
        waiting_id: str,
    ) -> None:
        """Builds the validator for an answer to one waiting call.

        Args:
            outputs: The node's outputs, by name. The answer fills them.
            choices: The choices the call's ``Ask`` offered, by output name.
            defaults: The defaults the call's ``Ask`` proposed, by output name.
            loc: Where the waiting call is in the run. The problems name it.
            waiting_id: The id of the waiting call being answered. The problems name it too.
        """
        self._outputs = outputs
        self._groups = exclusive_groups(outputs)
        self._choices = choices
        self._defaults = defaults
        self._loc = loc
        self._waiting_id = waiting_id

    def __call__(self, answer: Any) -> dict[str, Any]:
        """Checks ``answer`` and returns the outputs it fills.

        Args:
            answer: A plain value for a node with one question, or a mapping by question name for several.

        Returns:
            The node's outputs, by name, in the form the run keeps.

        Raises:
            InvalidCallError: See the class.
        """
        read, faults = self._read(answer)
        if faults:
            raise InvalidCallError(
                call_problems.invalid_answer(
                    *self._loc,
                    waiting_id=self._waiting_id,
                    reason=reason,
                    question=question or None,
                )
                for question, reason in faults
            )
        return read

    def _read(self, answer: Any) -> tuple[dict[str, Any], list[tuple[str | None, str]]]:
        """Reads ``answer`` into outputs, and collects every fault rather than stopping at the first.

        A question left out of the answer gets its default. With no default, it is a branch not chosen if it isn't
        required. Otherwise leaving it out is a fault.

        Returns:
            The outputs read, by name, and the faults. Each fault is ``(question, reason)``. ``question`` is
            ``None`` when the fault is with the answer as a whole.
        """
        outputs = self._outputs
        if len(outputs) == 1:
            ((name, field),) = outputs.items()
            try:
                return {name: _answer_value(field, answer, self._choices.get(name))}, []
            except PydanticCustomError as refused:
                return {}, [(None, refused.message())]
        if answer is None and not outputs:
            # A node with no outputs is answered with nothing. The person only says to go on.
            return {}, []
        if not isinstance(answer, Mapping):
            return {}, [
                (
                    None,
                    f"the answer isn't a mapping of {comma_separated(outputs, empty='nothing')}, one value per question",
                )
            ]
        faults: list[tuple[str | None, str]] = [
            (str(name), f"there is no question '{name}'") for name in answer if name not in outputs
        ]
        read: dict[str, Any] = {}
        for name, field in outputs.items():
            try:
                if name in answer:
                    read[name] = _answer_value(field, answer[name], self._choices.get(name))
                    continue
                question = as_asked(field, self._choices, self._defaults)
                if question.default is not MISSING:
                    read[name] = _read_as_answer(field.whole, question.default)
                elif not question.required:
                    read[name] = SKIPPED
                else:
                    faults.append((name, f"'{name}' has no answer"))
            except PydanticCustomError as refused:
                faults.append((name, f"'{name}': {refused.message()}"))
        for fault in groups_without_one_value(self._groups, answer):
            faults.append(
                (None, f"exactly one of {comma_separated(fault.members)} needs an answer")
            )
        return read, faults


def _answer_value(field: Field, value: Any, choices: tuple[Choice, ...] | None) -> Any:
    """Reads a person's answer to one question, and checks it against the choices.

    The output's type reads the answer first. That also checks any choices written on the type. Then
    ``_check_picks`` checks what the type can't: the ``Ask``'s own choices, and that a ``Series`` answer picks each
    choice only once.

    Say the question is about a ``str`` output, and the ``Ask`` offered ``Choice("ida", "Ida B.")`` and
    ``Choice("bo", "Bo K.")``. The answer ``"bo"`` is read as ``"bo"``. The answer ``"al"`` is refused, since it
    isn't one of the choices.

    Args:
        field: The output the question is about.
        value: The person's answer to it.
        choices: The choices the ``Ask`` offered, each id in its JSON form, or ``None`` if it offered none.

    Returns:
        The answer in the form the run keeps.

    Raises:
        PydanticCustomError: The answer isn't a value of the output's type, isn't one of the choices, or picks one
            choice twice. The message says which, without quoting the answer.
    """
    read = _read_as_answer(field.whole, value)
    if _has_picks_to_check(field, choices):
        _check_picks(field, field.whole.serializer.to_python(read, mode="json"), choices)
    return read


def _check_picks(field: Field, written: Any, choices: tuple[Choice, ...] | None) -> None:
    """Checks the picks in an answer that the output's type can't check by itself.

    There are two such checks. If the ``Ask`` offered choices, each pick must be one of them. And an answer to a
    ``Series`` must not pick the same choice twice. Only a person's answer is held to that: a node may return a
    ``Series`` that has a value twice.

    Picks are compared by their JSON text, the same rule a ``Param``'s choices hold an input or a return to. So
    ``1`` picks a ``float`` choice ``1.0``. ``None`` counts as no pick.

    Args:
        field: The output the answer is for.
        written: The answer after the output's type has read it, in its JSON form.
        choices: The choices the ``Ask`` offered, or ``None``.

    Raises:
        PydanticCustomError: ``not_a_choice`` if a pick isn't one of the ``Ask``'s choices, else ``picked_twice``
            if a choice is picked twice. The message doesn't quote the answer.
    """
    if not _has_picks_to_check(field, choices):
        return
    picks = pick_texts(written, several=field.is_series)
    if choices is not None:
        check_one_of(choice_texts(choices), picks)
    if len(set(picks)) != len(picks):
        raise PydanticCustomError("picked_twice", "a choice is picked twice")


def _has_picks_to_check(field: Field, choices: tuple[Choice, ...] | None) -> bool:
    """Whether ``_check_picks`` has anything to check: the ``Ask`` offered choices, or the output is a ``Series``
    whose items have choices."""
    return choices is not None or (field.is_series and field.type.choices is not None)


def _read_as_answer(value_type: Type, value: Any) -> Any:
    """Reads one value as a person's answer of type ``value_type``.

    An answer is read strictly, the way its JSON text would be read. So ``True`` is a ``bool``, but ``"yes"`` is
    refused. A ``date`` may be given as a ``date`` or as its ISO text, like ``"2026-11-15"``. Either way, an answer
    given from Python and the same answer sent over HTTP read the same. An ``Ask``'s default and choices are read
    by the same rules.

    Returns:
        The value in the form the run keeps. A ``Series`` answer is kept the way the run keeps a ``Series`` that a
        node returned.

    Raises:
        PydanticCustomError: ``no_json_form`` if the value can't be written as JSON. ``not_its_type`` if the type
            refuses it, with the type's reason as the message, quoting nothing of the value.
    """
    try:
        return validate_as_json(
            schema_output_validator_of(value_type), value, fallback=untyped_as_json
        )
    except PydanticSerializationError:
        raise PydanticCustomError("no_json_form", "the value can't be written as JSON") from None
    except PydanticCustomError as refused:  # a table with a column a saved run can't keep as it is
        raise PydanticCustomError("not_its_type", refused.message()) from None
    except ValidationError as error:
        raise PydanticCustomError("not_its_type", value_type.refusal(error).reason) from None
