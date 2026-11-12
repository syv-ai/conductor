"""The door for what a node raises: an ``Ask``'s choices and default, checked when it is raised.

Example: ``assign(ticket: str) -> str`` raises ``Ask("Who takes it?", choices=[Choice("ida", "Ida B."), "bo"])``.
The door returns the call's ``Question``, whose overrides for ``result`` are the two choices as records, each id in
its JSON form. Given ``choices=[1, 2]`` it raises ``RefusedError`` with ``invalid_ask``: a choice is what an answer
picks, so each is read as an answer is, and ``1`` is not a ``str``. An ``Ask`` raised by the function of an
``Asked`` parameter is checked against that parameter in the same way. A default is held to every rule an answer is,
the ``Ask``'s own choices included.

A ``NodeError``'s ``ctx`` is the other value a node raises; it is read where it is made, by ``NodeError`` itself, so
a ``ctx`` that is not JSON fails at the node's own ``raise``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import JsonValue
from pydantic_core import PydanticCustomError

from conductor._internal.ask import NOTHING_SET, Ask, Question, QuestionOverrides
from conductor._internal.field import Field
from conductor._internal.gate.answer import answer_value, from_json
from conductor._internal.gate.refused import RefusedError
from conductor._internal.markers import REQUIRED
from conductor._internal.problem import catalogue as problem
from conductor._internal.text import comma_separated
from conductor._internal.value_type import Choice


class AskDoor:
    """Checks an ``Ask`` raised for a node against the fields its answer fills (the node's outputs, or its
    ``Asked`` parameter), and keeps it as the call's question. Built once per node.

    Raises (when called):
        RefusedError: The ``Ask``'s choices or default don't fit the fields (``invalid_ask``, whose ``reason``
            says which part, quoting no value).
    """

    def __init__(self, fields: Mapping[str, Field]) -> None:
        """Builds the door for a node whose answer fills ``fields``, by name."""
        self._outputs = fields

    def __call__(self, ask: Ask, inputs: dict[str, Any]) -> Question:
        """Returns the question the call keeps until it is answered: the ``Ask``'s prompt, what the node received,
        and what the ``Ask`` set on each field's question.

        Args:
            ask: What the node raised.
            inputs: What the node received, as JSON.

        Raises:
            RefusedError: See the class.
        """
        if ask.choices is None and ask.default is REQUIRED:
            return Question(ask.prompt, inputs, NOTHING_SET)
        outputs = self._outputs
        overrides: dict[str, QuestionOverrides] = {}
        if ask.choices is not None:
            for name, choices in self._by_output(ask.choices, "choices").items():
                overrides[name] = {"choices": _offered(outputs[name], choices)}
        if ask.default is not REQUIRED:
            for name, default in self._by_output(ask.default, "default").items():
                overrides.setdefault(name, {})["default"] = _proposed(outputs[name], default, overrides.get(name, {}).get("choices"))
        return Question(ask.prompt, inputs, overrides)

    def _by_output(self, given: Any, argument: str) -> dict[str, Any]:
        """One ``Ask`` argument by output name: a lone output's as itself, several outputs' from the mapping given.

        Raises:
            RefusedError: The argument doesn't have the answer's shape, or names an output the node doesn't have
                (``invalid_ask``).
        """
        outputs = self._outputs
        if len(outputs) == 1:
            if argument == "choices" and isinstance(given, Mapping):
                raise RefusedError([problem.invalid_ask(reason="the node asks one question, so it is answered with a value, and choices= is a list")])
            return {next(iter(outputs)): given}
        if not isinstance(given, Mapping):
            reason = f"the node has {len(outputs)} outputs, so it is answered by output name, and {argument}= goes by output name too"
            raise RefusedError([problem.invalid_ask(reason=reason)])
        for name in given:
            if name not in outputs:
                reason = f"the {argument} names '{name}', which is not an output; the outputs are {comma_separated(outputs)}"
                raise RefusedError([problem.invalid_ask(reason=reason)])
        return dict(given)


def _offered(field: Field, choices: list[Any]) -> list[dict[str, Any]]:
    """The choices an ``Ask`` offers for one output, as records with their ids in JSON form.

    A choice is what an answer picks, so its id is read as an answer is; a ``Series``' ids each as one item.

    Raises:
        RefusedError: No choice is offered, a choice is offered twice, or one is not a value of the output's type
            (``invalid_ask``).
    """
    if not choices:
        raise RefusedError([problem.invalid_ask(reason=f"'{field.name}' is offered no choices")])
    given = [choice if isinstance(choice, Choice) else Choice(choice, str(choice)) for choice in choices]
    try:
        shown = [{"id": field.type.to_json(from_json(field.type, choice.id)), **choice.model_dump(mode="json", exclude={"id"})} for choice in given]
    except PydanticCustomError as refused:
        raise RefusedError([problem.invalid_ask(reason=f"a choice for '{field.name}' is not what it returns: {refused.message()}")]) from None
    if len({repr(choice["id"]) for choice in shown}) != len(shown):
        raise RefusedError([problem.invalid_ask(reason=f"a choice for '{field.name}' is offered twice")])
    return shown


def _proposed(field: Field, default: Any, choices: list[dict[str, Any]] | None) -> JsonValue:
    """The default an ``Ask`` proposes for one output, in JSON form, held to the rules an answer is.

    Args:
        field: The output.
        default: The proposed answer as given.
        choices: The choices the same ``Ask`` offers for the output, if any.

    Raises:
        RefusedError: An answer could not be the default (``invalid_ask``).
    """
    try:
        written: JsonValue = field.whole.to_json(answer_value(field, default, choices))
    except PydanticCustomError as refused:
        raise RefusedError([problem.invalid_ask(reason=f"the default for '{field.name}' is not an answer it takes: {refused.message()}")]) from None
    return written
