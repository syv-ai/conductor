"""Asking a person: ``raise Ask(...)`` in a node, the questions a host shows, and the answer read back.

The node's return type is what the person answers, and it says how to ask:
``-> str`` is a line of free text, ``-> bool`` a yes or no, and a frozen
dataclass a form of one question per field. How each question looks (its
title, its widget, choices known when the node is written) is the type's,
through ``Param``. ``Ask`` adds only what the run knows: the prompt, choices
known only when the node runs, and a proposed answer. Every return type's
question is in the reference, ``python -m conductor.about legs-and-asking``.

A waiting unit is a ``Waiting``, and each of its ``questions`` is the field
record ``describe()`` gives an output, with what the ``Ask`` set: so a host
draws a question with the code that draws an input. An answer is checked
against that record and the node's types before anything runs.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, TypedDict

import pydantic_core
from pydantic import JsonValue

from conductor._internal.address import Loc, loc_text
from conductor._internal.canonical_json import canonical_json
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.field import Field, FieldDescription
from conductor._internal.sentinels import REQUIRED, SKIPPED
from conductor._internal.value_type import Choice, FieldValueError

if TYPE_CHECKING:
    from conductor._internal.compiled import CompiledFunctionNode
    from conductor._internal.outcomes import Asking


class Ask(BaseException):
    """Stops a node's call to ask a person; the node's return type is what they answer.

    Nothing that reads the call runs until an answer arrives. The answer is the
    node's outputs, and the node is not called again for it. A
    free-text question is the plain case: ``-> str`` and ``raise Ask(prompt)``::

        @node("reason", title="Reason", description="Asks why a ticket was closed.")
        def reason(ticket: str) -> str:
            raise Ask(f"Why was {ticket} closed?")

        @node("assign", title="Assign", description="Asks who takes a ticket.")
        def assign(ticket: str) -> str:
            raise Ask(f"Who takes {ticket}?", choices=[Choice("ida", "Ida B."), Choice("bo", "Bo K.")])

    ``Ask`` derives from ``BaseException``, so an ``except Exception`` in the
    node lets it through.

    Args:
        prompt: What the person reads.
        choices: The values the answer is limited to, known only when the node
            runs, each a value or a ``Choice`` to give it a title. For a node
            with one output, a list; a ``Series`` output picks several of
            them. For a node with several outputs, which is answered by output
            name, a mapping by output name that names only the outputs it
            limits. Choices known when the node is written go on the type:
            a ``Literal``, or ``Param(choices=)``.
        default: The proposed answer, in the answer's shape: a value for a
            node with one output, a mapping by output name for several, naming
            only the outputs it proposes for. A question with a default may be
            left out of an answer. Left out, nothing is proposed beyond the
            defaults of a returned dataclass's fields.

    Attributes:
        prompt: What the person reads.
        choices: The choices as given, ``None`` when none were.
        default: The proposed answer as given, ``REQUIRED`` when none was.
    """

    def __init__(self, prompt: str, *, choices: list[Any] | Mapping[str, list[Any]] | None = None, default: Any = REQUIRED) -> None:
        super().__init__(prompt)
        self.prompt = prompt
        self.choices = choices
        self.default = default


class InvalidAskError(Exception):
    """An ``Ask`` that the node's outputs can't be asked with; the message says why and quotes no value.

    The leg turns it into the node's ``invalid_ask`` problem, the message as ``ctx.reason``.
    """


class QuestionOverrides(TypedDict, total=False):
    """What an ``Ask`` sets on one output's question, as JSON; a key is present only when the ``Ask`` set it.

    Attributes:
        choices: The choices offered, each its ``{"id", "title", ...}`` record with the id in its JSON form.
        default: The proposed answer in its JSON form.
    """

    choices: list[dict[str, Any]]
    default: JsonValue


#: The overrides of an ``Ask`` that set nothing, shared by every call such an ``Ask`` stops.
_NOTHING_SET: Mapping[str, QuestionOverrides] = MappingProxyType({})


@dataclass(slots=True)
class Question:
    """What a node asked on one call, kept on its unit while it waits and after it is answered, and saved with the run.

    A plain record, made for every call that asks, so it costs what a dict would; a save reads it back through
    pydantic. Like the unit that holds it, it is never changed once made.

    Attributes:
        prompt: What the person reads.
        inputs: What the node received, as JSON.
        overrides: What the ``Ask`` set on each question, by output name; empty when it set nothing.
    """

    prompt: str
    inputs: dict[str, Any]
    overrides: Mapping[str, QuestionOverrides] = dataclasses.field(default_factory=dict)


class Waiting(ConductorModel):
    """A call of a node waiting for a person, as a host shows it.

    A node with one output is answered with its value, one with several with
    a mapping by output name.

    Attributes:
        id: What ``run.answer`` takes: the node's id, then ``@`` and each place
            of its item, then ``~`` and a fingerprint of the question asked
            there, ``"approve@1~3fa2c0d1"``. Inside an embedded graph, the
            embedding node and its item come first, ``"review@0/approve@1~3fa2c0d1"``.
            An answer for an id whose place has since come to hold another
            question is refused as ``answer_stale``.
        loc: Which unit it is, as a ``Problem``'s ``loc`` says it: the node
            that asks, then the place of its item in each Series the node
            runs per item of, ``("approve", 1)``; inside an embedded graph,
            the embedding node and its item first, ``("review", 0, "approve", 1)``.
        prompt: What the node's ``Ask`` said.
        questions: One per output: the ``FieldDescription`` its ``describe()``
            gives, with the ``choices`` and ``default`` the ``Ask`` set and
            ``required`` worked out from them.
        inputs: What the node received, as JSON.
    """

    id: str
    loc: tuple[str | int, ...]
    prompt: str
    questions: list[FieldDescription]
    inputs: dict[str, Any]


def question_asked(ask: Ask, outputs: Mapping[str, Field], inputs: dict[str, Any]) -> Question:
    """Checks an ``Ask`` against the node's outputs and keeps it, with what the node received, as the unit's question.

    Each choice and the default are read as an answer is, so a choice or a
    default the output's type refuses is refused here, when the node asks.

    Args:
        ask: What the node raised.
        outputs: The node's outputs, by name.
        inputs: What the node received, as JSON.

    Returns:
        The question the unit holds until it is answered.

    Raises:
        InvalidAskError: The ``Ask``'s choices or default don't fit the outputs.
    """
    if ask.choices is None and ask.default is REQUIRED:
        return Question(ask.prompt, inputs, _NOTHING_SET)
    overrides: dict[str, QuestionOverrides] = {}
    if ask.choices is not None:
        for name, choices in _by_output(ask.choices, outputs, "choices").items():
            overrides[name] = {"choices": _offered(outputs[name], choices)}
    if ask.default is not REQUIRED:
        for name, default in _by_output(ask.default, outputs, "default").items():
            overrides.setdefault(name, {})["default"] = _proposed(outputs[name], default, overrides.get(name, {}).get("choices"))
    return Question(ask.prompt, inputs, overrides)


def _by_output(given: Any, outputs: Mapping[str, Field], argument: str) -> dict[str, Any]:
    """One ``Ask`` argument by output name: a lone output's as itself, several outputs' from the mapping given.

    Raises:
        InvalidAskError: The argument doesn't have the answer's shape, or names an output the node doesn't have.
    """
    if len(outputs) == 1:
        if argument == "choices" and isinstance(given, Mapping):
            raise InvalidAskError("the node has one output, so it is answered with a value, and choices= is a list")
        return {next(iter(outputs)): given}
    if not isinstance(given, Mapping):
        raise InvalidAskError(f"the node has {len(outputs)} outputs, so it is answered by output name, and {argument}= goes by output name too")
    for name in given:
        if name not in outputs:
            raise InvalidAskError(f"the {argument} names '{name}', which is not an output; the outputs are {', '.join(outputs) or 'none'}")
    return dict(given)


def _offered(field: Field, choices: list[Any]) -> list[dict[str, Any]]:
    """The choices an ``Ask`` offers for one output, as records with their ids in JSON form.

    A choice is what an answer picks, so its id is read as an answer is; a ``Series``' ids as one pick of them all.

    Raises:
        InvalidAskError: The output is a table, no choice is offered, a choice is offered twice, or one is not a
            value of the output's type.
    """
    if not choices:
        raise InvalidAskError(f"'{field.name}' is offered no choices")
    try:
        shown = [
            (choice if isinstance(choice, Choice) else Choice(choice, str(choice))).model_dump(mode="json")
            | {"id": field.type.to_json(field.type.read_json(choice.id if isinstance(choice, Choice) else choice))}
            for choice in choices
        ]
    except FieldValueError as refused:
        raise InvalidAskError(f"a choice for '{field.name}' is not what it returns: {refused.reason}") from None
    if len({repr(choice["id"]) for choice in shown}) != len(shown):
        raise InvalidAskError(f"a choice for '{field.name}' is offered twice")
    return shown


def _proposed(field: Field, default: Any, offered: list[dict[str, Any]] | None) -> JsonValue:
    """The default an ``Ask`` proposes for one output, in JSON form, held to the rules an answer is.

    Args:
        field: The output.
        default: The proposed answer as given.
        offered: The choices the same ``Ask`` offers for the output, if any.

    Raises:
        InvalidAskError: An answer could not be the default.
    """
    try:
        proposed: JsonValue = field.whole.to_json(_checked_answer(field, default, offered))
    except FieldValueError as refused:
        raise InvalidAskError(f"the default for '{field.name}' is not an answer it takes: {refused.reason}") from None
    return proposed


def output_questions(outputs: Mapping[str, Field]) -> dict[str, FieldDescription]:
    """Each output's question as the node's types set it, before an ``Ask`` sets anything.

    Args:
        outputs: The node's outputs, by name.

    Returns:
        A question per output, by name, under the output's own label, and a field of a returned dataclass with a
        default proposes it.
    """
    return {name: _question(field) for name, field in outputs.items()}


def _question(field: Field) -> FieldDescription:
    """One output's question as its type sets it."""
    question = field.description
    return question.model_copy(update={"required": _required(field, {}, limited=question.choices is not None)})


def waiting_questions(compiled_node: CompiledFunctionNode, overrides: Mapping[str, QuestionOverrides]) -> list[FieldDescription]:
    """A waiting unit's questions: each output's question with what its ``Ask`` set.

    Args:
        compiled_node: The node that asks.
        overrides: What the ``Ask`` set, by output name.

    Returns:
        A question per output, in the outputs' order.
    """
    outputs = compiled_node.outputs
    return [
        _overridden(outputs[name], question, overrides[name]) if name in overrides else question
        for name, question in compiled_node.questions.items()
    ]


def _overridden(field: Field, question: FieldDescription, overrides: QuestionOverrides) -> FieldDescription:
    """One output's question with what its ``Ask`` set, and whether it is ``required`` worked out again from that."""
    choices = overrides.get("choices", question.choices)
    return question.model_copy(update={**overrides, "required": _required(field, overrides, limited=choices is not None)})


def _required(field: Field, overrides: QuestionOverrides, *, limited: bool) -> bool:
    """Whether an answer must give the question.

    It needn't when the question has a default (its field's, ``SKIPPED`` too, or one its ``Ask`` set), is one of an
    exclusive group, or its type takes a blank and no choices limit what it takes.
    """
    defaulted = "default" in overrides or field.default is not REQUIRED
    blank = not field.many and field.whole.optional and not limited
    return not defaulted and field.param.exclusive_group is None and not blank


def answer_outputs(compiled_node: CompiledFunctionNode, value: Any, overrides: Mapping[str, QuestionOverrides]) -> tuple[dict[str, Any], list[tuple[str | None, str]]]:
    """Reads a person's answer as the node's outputs.

    Args:
        compiled_node: The node that asked.
        value: The answer: a value for a node with one output, a mapping by output name for several.
        overrides: What the node's ``Ask`` set, by output name.

    Returns:
        The outputs by name, and every fault in the answer as ``(question, reason)``, the question ``None`` for a
        fault of the whole answer. Nothing of the value is quoted.
    """
    outputs = compiled_node.outputs
    if len(outputs) == 1:
        ((name, field),) = outputs.items()
        try:
            return {name: _checked_answer(field, value, overrides.get(name, {}).get("choices", compiled_node.questions[name].choices))}, []
        except FieldValueError as refused:
            return {}, [(None, refused.reason)]
    if value is None and not outputs:
        return {}, []  # a node that returns nothing is answered with nothing: the person only says to go on
    if not isinstance(value, Mapping):
        return {}, [(None, f"answer each question by name: a mapping of {', '.join(outputs) or 'nothing'}")]
    faults: list[tuple[str | None, str]] = [(str(name), f"there is no question '{name}'") for name in value if name not in outputs]
    read: dict[str, Any] = {}
    groups: dict[str, list[str]] = {}
    for name, field in outputs.items():
        question, override, group = compiled_node.questions[name], overrides.get(name, {}), field.param.exclusive_group
        offered = override.get("choices", question.choices)
        try:
            if name in value:
                read[name] = _checked_answer(field, value[name], offered)
            elif group is not None:
                read[name] = SKIPPED
            elif field.default is SKIPPED and "default" not in override:
                read[name] = SKIPPED  # left out, a field whose default is SKIPPED is a branch not taken
            elif not _required(field, override, limited=offered is not None):
                read[name] = field.whole.read_answer(override.get("default", question.default))  # a default, or a blank where the type takes one
            else:
                faults.append((name, f"'{name}' was not answered"))
        except FieldValueError as refused:
            faults.append((name, f"'{name}': {refused.reason}"))
        if group is not None:
            groups.setdefault(group, []).append(name)
    for members in groups.values():
        if sum(name in value for name in members) != 1:
            faults.append((None, f"answer exactly one of {', '.join(members)}"))
    return read, faults


def _checked_answer(field: Field, value: Any, offered: list[dict[str, Any]] | None) -> Any:
    """Reads one question's answer into the node's own form, held to the choices ``offered``.

    Raises:
        FieldValueError: The answer is not a value of the output's type, not one of the choices, or, for a pick of several,
            picks a choice twice.
    """
    read = field.whole.read_answer(value)
    if offered is not None:
        picked: Any = field.whole.to_json(read)
        picks = picked if field.many else [picked]
        ids = [choice["id"] for choice in offered]
        if any(item not in ids for item in picks):
            raise FieldValueError("the answer is not one of the choices")
        if len({repr(item) for item in picks}) != len(picks):
            raise FieldValueError("a choice is picked twice")
    return read


def waiting_record(compiled_node: CompiledFunctionNode, loc: Loc, asking: Asking) -> Waiting:
    """A waiting unit as a host shows it. Its id is where it is as text, then ``~`` and the fingerprint of the
    question asked there, as ``run.answer`` reads it.

    Args:
        compiled_node: The node that asks.
        loc: Where the unit is: the node and its item, behind those of each embedded graph's node around it.
        unit: The unit.
    """
    question = asking.question
    return Waiting(
        id=f"{loc_text(loc)}~{question_fingerprint(compiled_node, asking)}", loc=loc, prompt=question.prompt,
        questions=waiting_questions(compiled_node, question.overrides), inputs=question.inputs,
    )


def question_fingerprint(compiled_node: CompiledFunctionNode, asking: Asking) -> str:
    """Eight hex digits of which node asks, as which code, and what it received, so a waiting id stops matching when its place holds another question."""
    inputs = asking.question.inputs
    values = [inputs[name] for name in sorted(inputs)]  # saved, so the names may come back from a store in any order
    hashed = [compiled_node.id, compiled_node.reuse_digest, values]
    written = canonical_json(hashed) if {type(v) for v in values} & {dict, list} else pydantic_core.to_json(hashed)  # plain values have no keys to sort
    return hashlib.blake2b(written, digest_size=4).hexdigest()
