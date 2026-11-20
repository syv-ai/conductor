"""The checks on asking a person: one on the ``Ask`` a node raises, one on the answer a person gives, and one on
the reason a person gives when they decline to answer.

``AskValidator`` runs when a node raises ``Ask``. It makes sure the choices and the default fit the node's outputs.
Say ``assign(ticket: str) -> str`` raises ``Ask("Who?", choices=[1, 2])``. The answer has to be a ``str``, so no
one could ever pick ``1`` or ``2``. The ``Ask`` is refused with ``invalid_ask``.

``AnswerValidator`` runs when a person answers. It turns the answer into the node's outputs. Say
``approve(line: str) -> bool`` is waiting. The answer ``True`` gives the outputs ``{"result": True}``. The answer
``"yes"`` is refused with ``invalid_answer``, since a yes or no is ``True`` or ``False``, not a word that means one.

``AnswerAsJsonValidator`` turns an answer into the JSON data it is read from. A host that keeps an answer to give a
run later keeps that data.

``DeclineValidator`` runs when a person declines with ``run.with_decline``. Its reason is text, or ``None`` for none, so
``reason=123`` is refused with ``invalid_decline_reason``.

The answer and the ``Ask``'s choices and default are each checked by one pydantic schema, built from conductor's
own readers, so every fault is found in one pass. An answer is read strictly, the way its JSON text would be read.
That way an answer given from Python and the same answer sent over HTTP are read the same. It is read by the same
rules as a value the node returns, since the answer stands in for that value.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any, cast

from pydantic import MISSING, JsonValue, ValidationError
from pydantic_core import (
    CoreSchema,
    SchemaValidator,
    core_schema,
)
from pydantic_core.core_schema import TypedDictField

from conductor._internal.base import REQUIRED, SKIPPED, Loc, Validator, validate_as_json
from conductor._internal.field import Field, answer_by_output, outputs_schema, without_skipped
from conductor._internal.problem import (
    InvalidCallError,
    call_problems,
    conductor_error,
    run_problems,
)
from conductor._internal.table import json_fallback
from conductor._internal.types import (
    Choice,
    Type,
    TypeRefusal,
    check_one_of,
    choices_schema,
    pick_texts,
    read_as_argument,
    read_as_output,
    written_as_json,
)

from .question import as_asked
from .type import Ask

#: Explains a fault no output's type explains, such as an answer of several questions that isn't a mapping.
_ANY = Type(Any)


class AskValidator(Validator[Ask, tuple[dict[str, tuple[Choice, ...]], dict[str, JsonValue]]]):
    """A validator for the ``Ask`` a node raises.

    The answer to an ``Ask`` fills the node's outputs, so the ``Ask``'s choices and default must fit them. Its
    choices are read by each output's type, as a person's answer is. Its default is read as an answer would be,
    by the schema ``AnswerValidator`` reads by, so it must be an answer a person could give, one of the ``Ask``'s
    choices included. The validator also turns them into the form the waiting call keeps.

    One validator is built for each node. If the node doesn't say ``asks=True``, it refuses every ``Ask``.

    Raises (when called):
        InvalidCallError: ``ask_not_declared`` if the node doesn't say it asks. One ``invalid_ask`` with every fault
            of the choices and the default in ``ctx.errors``, each ``loc`` starting with ``choices`` or ``default``,
            then the output's name, then a choice's index. Its ``reason`` says which part is wrong first, without
            quoting the value.
    """

    def __init__(self, outputs: Mapping[str, Field], *, asks: bool) -> None:
        """Builds the validator for one node.

        Args:
            outputs: The node's outputs, by name.
            asks: Whether the node says it may ask, with ``asks=True``.
        """
        self._outputs = outputs
        self._asks = asks

    @functools.cached_property
    def _choices(self) -> SchemaValidator:
        """Checks an ``Ask``'s choices, by output name. Built when the node first asks.

        They have the answer's shape (``answer_by_output``): a node of one output is given its list alone, a node
        of several a mapping by output name, or a dataclass or a model of the outputs. On a ``Series`` output the
        choices are for its items, so each id is read as one item.
        """
        return SchemaValidator(
            core_schema.no_info_before_validator_function(
                functools.partial(answer_by_output, self._outputs),
                core_schema.typed_dict_schema(
                    {
                        name: core_schema.typed_dict_field(
                            choices_schema(
                                functools.partial(read_as_output, field.type),
                                field.type.to_json,
                                known=field.type.choices or (),
                            ),
                            required=False,
                        )
                        for name, field in self._outputs.items()
                    },
                    extra_behavior="forbid",
                ),
            )
        )

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
        faults: list[dict[str, Any]] = []
        if ask.choices is not None:
            try:
                choices = self._choices.validate_python(ask.choices)
            except ValidationError as error:
                faults += _ask_faults("choices", error, outputs, items=True)
        if ask.default is not REQUIRED:
            # A default is a ready-made answer, so it is read as an answer, against this Ask's own choices. It
            # proposes answers to some questions, so every question may be left out, and none has a default.
            fields = {
                name: core_schema.typed_dict_field(
                    _question(field, choices.get(name)), required=False
                )
                for name, field in outputs.items()
            }
            proposed = SchemaValidator(outputs_schema(outputs, fields, "answer"))
            try:
                read: dict[str, Any] = proposed.validate_python(ask.default)
            except ValidationError as error:
                faults += _ask_faults("default", error, outputs)
            else:
                defaults = {
                    name: outputs[name].whole.to_json(value)
                    for name, value in read.items()
                    if value is not SKIPPED
                }
        if faults:
            # each fault's loc is the argument, then the output it is for, if it is about one
            argument, *inside = faults[0]["loc"]
            for_output = f" for '{inside[0]}'" if inside and inside[0] in outputs else ""
            wrong = "are wrong" if argument == "choices" else "is wrong"
            raise InvalidCallError(
                [
                    run_problems.invalid_ask(
                        reason=f"the {argument}{for_output} {wrong} ({faults[0]['msg']})",
                        errors=faults,
                    )
                ]
            )
        return choices, defaults


def _ask_faults(
    argument: str, error: ValidationError, outputs: Mapping[str, Field], *, items: bool = False
) -> list[dict[str, Any]]:
    """Returns the faults in one argument of an ``Ask``, as ``ctx.errors`` holds them, each ``loc`` written from the
    argument down: ``["choices", "result", 1]``.

    A key of a mapping that isn't one of the node's outputs may be built from data, so it is never quoted: its fault
    is at ``"[key]"``, ``["choices", "[key]"]``, as a key that isn't text is named wherever conductor reads a mapping.

    Args:
        argument: ``"choices"`` or ``"default"``.
        error: What the argument's schema raised.
        outputs: The node's outputs, by name.
        items: Whether the argument is the choices, a list per output.
    """
    faults: list[dict[str, Any]] = []
    for place, refusal in _refusals(error, outputs, items=items):
        named = place if not place or str(place[0]) in outputs else ("[key]",)
        faults += [
            {**fault, "loc": [argument, *named, *fault["loc"]]} for fault in refusal["errors"]
        ]
    return faults


class AnswerValidator(Validator[Any, dict[str, Any]]):
    """A validator for a person's answer to a node that raised ``Ask``.

    It checks the answer and turns it into the node's outputs. A node with one output is answered with a plain
    value. A node with several is answered with a mapping by output name, one entry per question, or with a
    dataclass or a model of them. An entry may be left out when its question has a default, or when it is a branch,
    which is then skipped. The reference has the rules, under "Asking a person"
    (``python -m conductor.about asking-a-person``).

    One validator is built for each answer, from the questions the call asked.

    Raises (when called):
        InvalidCallError: Every fault at once, as one ``invalid_answer`` per question that is wrong, with
            ``ctx.question`` naming it on a node of several, and one for the answer as a whole: no mapping where
            there are several questions, or more than one branch answered. A name that isn't a question is its own,
            named by ``ctx.question``. Each has ``ctx.errors``, with pydantic's codes and no part of the answer.
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
        self._schema = SchemaValidator(
            outputs_schema(outputs, _answer_fields(outputs, choices, defaults), "answer")
        )
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
        try:
            read: dict[str, Any] = self._schema.validate_python(answer)
        except ValidationError as error:
            raise _refused_answer(error, self._outputs, self._loc, self._waiting_id) from None
        return read


def _refused_answer(
    error: ValidationError, outputs: Mapping[str, Field], loc: Loc, waiting_id: str
) -> InvalidCallError:
    """Turns what an answer's schema refused into one ``invalid_answer`` per place, as ``AnswerValidator`` raises it.

    Args:
        error: What the schema raised, each ``loc`` starting with an output's name.
        outputs: The node's outputs, by name.
        loc: Where the waiting call is in the run.
        waiting_id: The id of the waiting call being answered.
    """
    several = len(outputs) != 1
    return InvalidCallError(
        call_problems.invalid_answer(
            *loc,
            waiting_id=waiting_id,
            question=str(place[0]) if place and several else None,
            **refusal,
        )
        for place, refusal in _refusals(error, outputs)
    )


def _answer_fields(
    outputs: Mapping[str, Field],
    choices: Mapping[str, tuple[Choice, ...]],
    defaults: Mapping[str, JsonValue],
) -> dict[str, TypedDictField]:
    """Returns how a person's answer to each question of one call is read, by output name.

    Each question is read by ``_question``. A question is required as the host is shown it (``as_asked``). One left
    out of the answer gets its default, and one with no default that isn't required, a branch, is ``SKIPPED``.

    Args:
        outputs: The node's outputs, by name.
        choices: The choices the ``Ask`` offered, by output name.
        defaults: The defaults the ``Ask`` proposed, by output name.
    """
    questions = as_asked(outputs, choices, defaults)
    fields: dict[str, TypedDictField] = {}
    for name, field in outputs.items():
        schema = _question(field, choices.get(name))
        default, required = questions[name].default, questions[name].required
        # a required question's default is only what a host starts the answer with, as on a node of one output
        if not required and default is not MISSING:
            schema = core_schema.with_default_schema(
                schema,
                default_factory=functools.partial(read_as_output, field.whole, default),
            )
        fields[name] = core_schema.typed_dict_field(schema, required=required)
    return fields


def _question(field: Field, choices: tuple[Choice, ...] | None) -> CoreSchema:
    """Builds the schema of one question: a person's answer read as the value of its output.

    The answer is read strictly, the way its JSON text would be (``read_as_output``). So ``True`` is a ``bool``, but
    ``"yes"`` is refused. A ``date`` may be given as a ``date`` or as its ISO text, like ``"2026-11-15"``. The
    output's type checks any choices written on it. Then ``_check_picks`` checks what the type can't.

    Say the question is about a ``str`` output, and the ``Ask`` offered ``Choice("ida", "Ida B.")`` and
    ``Choice("bo", "Bo K.")``. The answer ``"bo"`` is read as ``"bo"``. The answer ``"al"`` is refused, since it
    isn't one of the choices.

    Args:
        field: The output the question is about.
        choices: The choices the ``Ask`` offered, each id in its JSON form, or ``None`` if it offered none.

    Returns:
        A schema whose value is the answer in the form the run keeps. A ``Series`` answer is kept the way the run
        keeps a ``Series`` that a node returned.
    """
    whole = field.whole

    def read(value: Any) -> Any:
        answer = read_as_output(whole, value)
        _check_picks(field, answer, choices)
        return answer

    return core_schema.no_info_plain_validator_function(read)


def _check_picks(field: Field, answer: Any, choices: tuple[Choice, ...] | None) -> None:
    """Checks the picks in an answer that the output's type can't check by itself.

    There are two such checks. If the ``Ask`` offered choices, each pick must be one of them. And an answer to a
    ``Series`` whose items have choices must not pick the same choice twice. Only a person's answer is held to
    that: a node may return a ``Series`` that has a value twice.

    Picks are compared by their JSON text, the same rule a ``Param``'s choices hold an input or a return to. So
    ``1`` picks a ``float`` choice ``1.0``. ``None`` counts as no pick.

    Args:
        field: The output the answer is for.
        answer: The answer after the output's type has read it.
        choices: The choices the ``Ask`` offered, or ``None``.

    Raises:
        PydanticCustomError: ``not_a_choice`` if a pick isn't one of the ``Ask``'s choices, else ``picked_twice``
            if a choice is picked twice. The message doesn't quote the answer.
    """
    if choices is None and not (field.is_series and field.type.choices is not None):
        return
    picks = pick_texts(
        field.whole.serializer.to_python(answer, mode="json"), several=field.is_series
    )
    if choices is not None:
        check_one_of(choices, picks)
    if len(set(picks)) != len(picks):
        raise conductor_error("picked_twice", "a choice is picked twice")


def _refusals(
    error: ValidationError, outputs: Mapping[str, Field], *, items: bool = False
) -> list[tuple[tuple[str | int, ...], TypeRefusal]]:
    """Splits what a schema of this module refused by the place each fault is at, and says why without the value.

    A place is an output's name, and for an ``Ask``'s choices a choice's index after it. Its faults are explained by
    the output's type (``Type.refusal``), which keeps only the steps inside the value its type declares. A fault at
    any other place is explained as it is: one of the whole (``()``), one at a name that isn't an output, and one of
    a list of choices as a whole.

    Args:
        error: What the schema raised, each ``loc`` starting with an output's name.
        outputs: The node's outputs, by name.
        items: Whether the schema read an ``Ask``'s choices, a list per output.

    Returns:
        Each place, in the order of its first fault, with its faults.
    """
    by_place: dict[tuple[str | int, ...], Type | None] = {}
    for raw in error.errors():
        loc = raw["loc"]
        field = outputs.get(str(loc[0])) if loc else None
        if field is None:
            by_place.setdefault(loc, None)
        elif not items:
            by_place.setdefault(loc[:1], field.whole)
        elif len(loc) > 1 and isinstance(loc[1], int):
            by_place.setdefault(loc[:2], field.type)
        else:
            by_place.setdefault(loc, None)
    return [
        (
            place,
            value_type.refusal(error, under=place)
            if value_type
            else _ANY.refusal(error, under=place, exactly=True),
        )
        for place, value_type in by_place.items()
    ]


class AnswerAsJsonValidator(Validator[Any, Any]):
    """A validator that turns a person's answer into the JSON data it is read from.

    An answer is read from its JSON form. A host that keeps an answer to give a run later keeps what this returns
    (``graph.answers_as_json``): a ``date`` becomes ``"2026-11-15"``, a dataclass or a pydantic model a mapping of its
    fields, and a ``polars.DataFrame`` its columns (``json_fallback``), since the question isn't known here to read
    the frame as it is. ``SKIPPED`` on an output is read by the rule a run reads it by (``without_skipped``): a branch
    not chosen is left out, and on an output that isn't a branch it is refused. So a kept answer reads as the answer
    itself, but for a frame with a column a saved run can't keep, which is refused here even where the question's
    table doesn't name it. ``inf`` stays a float, so the answer is refused where the run reads it
    (``finite_number``), as the answer itself is.

    Raises (when called):
        InvalidCallError: ``invalid_answer`` for ``SKIPPED`` on an output that isn't a branch, at its question, as
            ``run.with_answers`` refuses it. One ``invalid_answer`` for the whole answer when it can't be written as
            JSON, such as a ``str`` with a lone surrogate, or has a set in it (``set_has_no_order``). Its question
            isn't read here, so it names none.
    """

    def __init__(self, outputs: Mapping[str, Field] | None, *, loc: Loc, waiting_id: str) -> None:
        """Builds the validator for an answer to one waiting call.

        Args:
            outputs: The outputs of the node the waiting id names, by name, or ``None`` when the graph has no such
                node. Then the run refuses the answer as ``not_waiting``, and it is only written as JSON here.
            loc: The call the waiting id names. The problem names it.
            waiting_id: The id the answer was given for. The problem names it too.
        """
        self._outputs = outputs
        self._loc = loc
        self._waiting_id = waiting_id

    def __call__(self, answer: Any) -> Any:
        """Returns ``answer`` as the JSON data it is read from.

        Raises:
            InvalidCallError: See the class.
        """
        if self._outputs is not None:
            answer = self._without_skipped(self._outputs, answer)
        try:
            return written_as_json(
                _ANY.id,
                answer,
                lambda: validate_as_json(_ANY.output_validator, answer, fallback=json_fallback),
            )
        except ValidationError as error:
            raise InvalidCallError(
                [
                    call_problems.invalid_answer(
                        *self._loc, waiting_id=self._waiting_id, **_ANY.refusal(error, exactly=True)
                    )
                ]
            ) from None

    def _without_skipped(self, outputs: Mapping[str, Field], answer: Any) -> Any:
        """Returns the answer with each branch that is ``SKIPPED`` left out, by the rule the run reads it by.

        Raises:
            InvalidCallError: ``invalid_answer`` for each output that is ``SKIPPED`` but isn't a branch.
        """
        given = answer_by_output(outputs, answer)
        if not isinstance(given, Mapping):
            return answer
        kept, refused = without_skipped(outputs, given)
        if refused:
            raise _refused_answer(
                ValidationError.from_exception_data("answer", refused),
                outputs,
                self._loc,
                self._waiting_id,
            )
        # a node of one output is answered with its value, and a node of none with None
        return answer if len(outputs) < 2 else kept


#: The type of a decline's reason: a ``str``, or ``None``.
_REASON = Type(str | None)


class DeclineValidator(Validator[Any, str | None]):
    """A validator for the reason a person gives when they decline to answer a node that raised ``Ask``.

    The reason is text, or ``None`` for none. ``run.with_decline`` puts it in the ``ctx`` of the run's
    ``answer_declined`` problem, which a saved run writes as JSON, so it is checked as its JSON text would be, as an
    answer is.
    ``"Over budget"`` is taken. ``123`` is refused, and so is a ``str`` JSON can't write, such as a lone surrogate.

    Raises (when called):
        InvalidCallError: One ``invalid_decline_reason`` problem, with pydantic's errors. It doesn't quote the
            reason.
    """

    def __init__(self, *, loc: Loc, waiting_id: str) -> None:
        """Builds the validator for a decline of one waiting call.

        Args:
            loc: Where the waiting call is in the run. The problem names it.
            waiting_id: The id the decline was given for. The problem names it too.
        """
        self._loc = loc
        self._waiting_id = waiting_id

    def __call__(self, reason: Any) -> str | None:
        """Checks ``reason`` and returns it.

        Raises:
            InvalidCallError: See the class.
        """
        try:
            return cast(str | None, read_as_argument(_REASON, reason))
        except ValidationError as error:
            raise InvalidCallError(
                [
                    call_problems.invalid_decline_reason(
                        *self._loc,
                        waiting_id=self._waiting_id,
                        errors=_REASON.refusal(error)["errors"],
                    )
                ]
            ) from None
