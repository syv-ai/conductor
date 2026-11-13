"""The door for what a node raises: an ``Ask``'s choices and default, checked when it is raised.

Example: ``assign(ticket: str) -> str`` raises ``Ask("Who takes it?", choices=[Choice("ida", "Ida B."), "bo"])``.
The door returns the call's ``Question``, whose overrides for ``result`` are the two choices as the output's type
reads them (``read_choices`` with the gate's validator of the output), each id in its JSON form. Given ``choices=[1, 2]`` it raises ``InvalidCallError``
with ``invalid_ask``: a choice is what an answer picks, so each is read strictly from its JSON form, and ``1`` is
not a ``str``.
A default is held to every rule an answer is, the ``Ask``'s own choices included.

A ``NodeError``'s ``ctx`` is the other value a node raises; it is read where it is made, by ``NodeError`` itself, so
a ``ctx`` that is not JSON fails at the node's own ``raise``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import JsonValue
from pydantic_core import PydanticCustomError

from conductor._internal.ask import NOTHING_SET, Ask, Question, QuestionOverrides
from conductor._internal.base import REQUIRED, comma_separated
from conductor._internal.field import Field
from conductor._internal.problem import InvalidCallError, run_problems
from conductor._internal.types import Choice, read_choices

from .answer import answer_value
from .rules import output_validator_of


class AskDoor:
    """Checks an ``Ask`` raised for a node against the node's outputs, which its answer fills, and keeps it as the
    call's question. Built once per node.

    Raises (when called):
        InvalidCallError: The ``Ask``'s choices or default don't fit the outputs (``invalid_ask``, whose ``reason``
            says which part, quoting no value).
    """

    def __init__(self, outputs: Mapping[str, Field]) -> None:
        """Builds the door for a node with ``outputs``, by name."""
        self._outputs = outputs

    def __call__(self, ask: Ask, inputs: dict[str, Any]) -> Question:
        """Returns the question the call keeps until it is answered: the ``Ask``'s prompt, what the node received,
        and what the ``Ask`` set on each output's question.

        Args:
            ask: What the node raised.
            inputs: What the node received, as JSON.

        Raises:
            InvalidCallError: See the class.
        """
        if ask.choices is None and ask.default is REQUIRED:
            return Question(ask.prompt, inputs, NOTHING_SET)
        outputs = self._outputs
        overrides: dict[str, QuestionOverrides] = {}
        if ask.choices is not None:
            for name, choices in self._by_output(ask.choices, "choices").items():
                value_type = outputs[name].type  # a Series' choices are its items', so each id is read as one item
                try:
                    overrides[name] = {"choices": read_choices(choices, output_validator_of(value_type), value_type.to_json)}
                except PydanticCustomError as refused:
                    raise InvalidCallError([run_problems.invalid_ask(reason=f"the choices for '{name}' are refused: {refused.message()}")]) from None
        if ask.default is not REQUIRED:
            for name, default in self._by_output(ask.default, "default").items():
                overrides.setdefault(name, {})["default"] = _proposed(outputs[name], default, overrides.get(name, {}).get("choices"))
        return Question(ask.prompt, inputs, overrides)

    def _by_output(self, given: Any, argument: str) -> dict[str, Any]:
        """One ``Ask`` argument by output name: a lone output's as itself, several outputs' from the mapping given.

        Raises:
            InvalidCallError: The argument doesn't have the answer's shape, or names an output the node doesn't have
                (``invalid_ask``).
        """
        outputs = self._outputs
        if len(outputs) == 1:
            if argument == "choices" and isinstance(given, Mapping):
                raise InvalidCallError([run_problems.invalid_ask(reason="the node asks one question, so it is answered with a value, and choices= is a list")])
            return {next(iter(outputs)): given}
        if not isinstance(given, Mapping):
            reason = f"the node has {len(outputs)} outputs, so it is answered by output name, and {argument}= goes by output name too"
            raise InvalidCallError([run_problems.invalid_ask(reason=reason)])
        for name in given:
            if name not in outputs:
                reason = f"the {argument} names '{name}', which is not an output; the outputs are {comma_separated(outputs)}"
                raise InvalidCallError([run_problems.invalid_ask(reason=reason)])
        return dict(given)


def undeclared_ask(ask: Ask, inputs: dict[str, Any]) -> Question:
    """The ``Ask`` door of a node that does not say it asks (no ``asks=True``): it refuses
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
        written: JsonValue = field.whole.to_json(answer_value(field, default, choices))
    except PydanticCustomError as refused:
        raise InvalidCallError([run_problems.invalid_ask(reason=f"the default for '{field.name}' is not an answer it takes: {refused.message()}")]) from None
    return written
