"""Asking a person: ``raise Ask(...)`` in a node, the questions a host shows, and the answer read back.

The node's return type is what the person answers, and it says how to ask:
``-> bool`` is a yes or no, and a frozen dataclass is a form of one question
per field. ``Ask`` adds only what the run knows: the prompt, the choices on
offer, a proposed answer and a control. Every return type's question is in
the reference, ``python -m conductor.about legs-and-asking``.

A waiting row is a ``Waiting``, and each of its ``questions`` is the field
record ``describe()`` gives an output, plus what the ``Ask`` said: so a host
draws a question with the code that draws an input. An answer is checked
against that record and the node's types before anything runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypedDict, Unpack

import polars as pl
from pydantic import JsonValue, TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError, to_json, to_jsonable_python

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import value_free_errors
from conductor._internal.fields import SKIPPED, FieldDescription, FieldSpec, Refused

if TYPE_CHECKING:
    from conductor._internal.graph import PlacedNode
    from conductor._internal.node_definition import NodeDefinition
    from conductor.widgets import Widget


class Choice(ConductorModel):
    """One option a person may pick: the ``id`` the answer holds and the ``title`` they read: ``Choice("ida", "Ida B.")``.

    A host subclasses it to carry more about an option; the fields it adds
    reach the question, and a dropdown, as written.
    """

    id: Any
    title: str

    def __init__(self, id: Any, title: str, **more: Any) -> None:  # noqa: A002  the word a host's option has
        fields: dict[str, Any] = {"id": id, "title": title, **more}  # the base's own signature names no fields
        super().__init__(**fields)


class AskArguments(TypedDict, total=False):
    """What ``Ask`` takes beside its prompt; see ``Ask``."""

    choices: Sequence[Any] | Mapping[str, Sequence[Any]]
    default: Any
    widget: Widget | Mapping[str, JsonValue]
    defaults: Mapping[str, Any]
    widgets: Mapping[str, Widget | Mapping[str, JsonValue]]


class Ask(BaseException):
    """Raised by a node to stop on its row and ask a person; the node's return type is what they answer.

    ``prompt`` is what the person reads. For a node with one output,
    ``choices`` limits the answer to these values (each a value, or a
    ``Choice`` to give it a title; a pick of several for a ``Series``),
    ``default`` proposes an answer and ``widget`` is a control of the host's,
    as on ``Param``. For a node with several outputs the same go by output
    name: ``choices`` a mapping, ``defaults`` and ``widgets``. A question with
    a default may be left out of the answer. Nothing runs for the row until an
    answer arrives; the answer is the node's outputs, and the node is not
    called again for that row::

        @node("assign")
        def assign(ticket: str) -> str:
            raise Ask(f"Who takes {ticket}?", choices=[Choice("ida", "Ida B."), Choice("bo", "Bo K.")])

    A ``BaseException``, so an ``except Exception`` in the node lets it through.
    """

    def __init__(self, prompt: str, **given: Unpack[AskArguments]) -> None:
        #: What the ``Ask`` adds to its questions, by argument; empty for a plain one, which costs nothing more.
        self.given: dict[str, Any] = dict(given)
        if not isinstance(prompt, str):
            raise TypeError(f"Ask's prompt is the text a person reads, not {type(prompt).__name__}")
        super().__init__(prompt)
        self.prompt = prompt


class Waiting(ConductorModel):
    """A row of a node waiting for a person, as a host shows it.

    ``id`` is what ``run.answer`` takes: the node's id, then ``@`` and each
    place of the row, then ``~`` and a fingerprint of the question asked there,
    ``"approve@1~3fa2c0d1"``; an answer for an id whose row has since come to
    hold another question is refused as ``answer_stale``. ``prompt`` is what the node's ``Ask``
    said; ``inputs`` is what the node received, as JSON. ``questions`` has one
    entry per output, the ``FieldDescription`` its ``describe()`` gives, with
    ``choices`` (from the ``Ask`` or a ``Literal``), ``default`` and ``widget``
    as the ``Ask`` set them and ``required`` worked out from them. A node with
    one output is answered with its value, one with several with a mapping by
    name.
    """

    id: str
    node_id: str
    row: tuple[int, ...]
    prompt: str
    questions: list[FieldDescription]
    inputs: dict[str, Any]


class WrongAsk(Exception):
    """A node raised ``Ask`` with something its outputs can't be asked with; the text says what, quoting no value."""


def asked(ask: Ask, definition: NodeDefinition[..., Any]) -> dict[str, dict[str, Any]]:
    """What ``ask`` adds to each question, by output name, as JSON; ``WrongAsk`` if it doesn't fit the node's outputs."""
    outputs = definition.outputs
    given = ask.given
    unknown = sorted(given.keys() - AskArguments.__annotations__.keys())
    if unknown:
        raise WrongAsk(f"Ask has no argument '{unknown[0]}'; it takes {', '.join(AskArguments.__annotations__)}")
    lone = next(iter(outputs)) if len(outputs) == 1 else None
    by_name = isinstance(given.get("choices"), Mapping)
    if lone is not None:
        if "defaults" in given or "widgets" in given or by_name:
            raise WrongAsk("the node has one output, so it takes default=, widget= and a list of choices")
        choices, defaults, widgets = ({lone: given[kind]} if kind in given else {} for kind in ("choices", "default", "widget"))
    else:
        if "default" in given or "widget" in given or ("choices" in given and not by_name):
            raise WrongAsk(f"the node has {len(outputs)} outputs, so it takes defaults=, widgets= and choices= by output name")
        choices, defaults, widgets = (dict(given.get(kind) or {}) for kind in ("choices", "defaults", "widgets"))
    added: dict[str, dict[str, Any]] = {}
    for kind, given in (("choices", choices), ("default", defaults), ("widget", widgets)):
        for name, value in given.items():
            spec = outputs.get(name)
            if spec is None:
                raise WrongAsk(f"the {kind} names '{name}', which is not an output; the outputs are {', '.join(outputs) or 'none'}")
            added.setdefault(name, {})[kind] = _added(spec, kind, value, added.get(name, {}))
    return added


def _added(spec: FieldSpec, kind: str, value: Any, so_far: Mapping[str, Any]) -> Any:
    """One thing an ``Ask`` adds to a question, as JSON; a default is held to the rules an answer is, against the choices ``so_far``."""
    if kind == "widget":
        try:
            return to_jsonable_python(value)
        except Exception:  # noqa: BLE001  a value with no JSON form, such as a class passed bare
            raise WrongAsk(f"the widget for '{spec.name}' has no JSON form; give an instance, such as Textarea()") from None
    if kind == "default":
        if value is SKIPPED:
            raise WrongAsk(f"the default for '{spec.name}' is SKIPPED; a person answers a value")
        try:
            return _as_json(spec, _checked_answer(spec, value, so_far))
        except Refused as refused:
            raise WrongAsk(f"the default for '{spec.name}' is not an answer it takes: {refused.reason}") from None
    if spec.shape == "table":
        raise WrongAsk(f"'{spec.name}' is a table, which takes no choices")
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise WrongAsk(f"the choices for '{spec.name}' are a list")
    ids = [choice.id if isinstance(choice, Choice) else choice for choice in value]
    try:  # a choice is what an answer picks, so it is read as an answer is: a Series' as one pick of them all
        kept = _as_json(spec, read_answer(spec, ids)) if spec.shape == "many" else [_as_json(spec, read_answer(spec, i)) for i in ids]
    except Refused as refused:
        raise WrongAsk(f"a choice for '{spec.name}' is not what it returns: {refused.reason}") from None
    offered = [{**choice.model_dump(mode="json"), "id": k} if isinstance(choice, Choice) else {"id": k, "title": str(k)} for choice, k in zip(value, kept, strict=True)]
    if not offered:
        raise WrongAsk(f"'{spec.name}' is offered no choices")
    if len({repr(choice["id"]) for choice in offered}) != len(offered):
        raise WrongAsk(f"a choice for '{spec.name}' is offered twice")
    return offered


def questions(placed: PlacedNode, added: Mapping[str, Mapping[str, Any]]) -> list[FieldDescription]:
    """A waiting row's questions: each output's record, with what its ``Ask`` added; one the ``Ask`` adds nothing to is
    the node's own record, made once."""
    if not added:
        return list(placed.question_records.values())
    return [FieldDescription.model_validate(question) if question["name"] in added else placed.question_records[question["name"]] for question in _asked(placed, added)]


def _asked(placed: PlacedNode, added: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``questions`` as JSON, with only the keys each question has."""
    out = []
    for name, question in placed.questions.items():
        extra = added.get(name)
        if extra:
            question = {**question, **extra}
            question["required"] = _required(placed.definition.outputs[name], question)
        else:
            question = dict(question)
        out.append(question)
    return out


def questions_of(definition: NodeDefinition[..., Any]) -> dict[str, dict[str, Any]]:
    """Each output's question before any ``Ask`` adds to it; a lone output takes the node's title."""
    lone = len(definition.outputs) == 1
    return {name: _question(spec, definition.title if lone else None) for name, spec in definition.outputs.items()}


def _question(spec: FieldSpec, node_title: str | None) -> dict[str, Any]:
    question = spec.as_json()
    if node_title is not None and spec.param.title is None:
        question["title"] = node_title  # a lone output's own name, "result", says nothing to a person
    if not spec.required:  # a field of a returned dataclass with a default proposes it
        question["default"] = _as_json(spec, spec.default)
    question["required"] = _required(spec, question)
    return question


def _required(spec: FieldSpec, question: Mapping[str, Any]) -> bool:
    """Whether an answer must give the question: not when it has a default or is one of an exclusive group, nor when its
    type takes a blank and no choices limit what it takes."""
    blank = spec.shape == "one" and spec.optional and "choices" not in question
    return "default" not in question and spec.param.exclusive_group is None and not blank


def answer_outputs(placed: PlacedNode, value: Any, added: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], list[tuple[str | None, str]]]:
    """A person's answer as the node's outputs, or every fault in it as ``(question, reason)``; nothing of the value is quoted."""
    outputs = placed.definition.outputs
    asked_now = {question["name"]: question for question in _asked(placed, added)}
    if len(outputs) == 1:
        (name, spec), = outputs.items()
        try:
            return {name: _checked_answer(spec, value, asked_now[name])}, []
        except Refused as refused:
            return {}, [(None, refused.reason)]
    if value is None and not outputs:
        return {}, []
    if not isinstance(value, Mapping):
        return {}, [(None, f"answer each question by name: a mapping of {', '.join(outputs) or 'nothing'}")]
    faults: list[tuple[str | None, str]] = [(str(name), f"there is no question '{name}'") for name in value if name not in outputs]
    read: dict[str, Any] = {}
    groups: dict[str, list[str]] = {}
    for name, spec in outputs.items():
        question, group = asked_now[name], spec.param.exclusive_group
        try:
            if name in value:
                read[name] = _checked_answer(spec, value[name], question)
            elif group is not None:
                read[name] = SKIPPED
            elif "default" in question:
                read[name] = read_answer(spec, question["default"])
            elif not question["required"]:
                read[name] = None
            else:
                faults.append((name, f"'{name}' was not answered"))
        except Refused as refused:
            faults.append((name, f"'{name}': {refused.reason}"))
        if group is not None:
            groups.setdefault(group, []).append(name)
    for members in groups.values():
        if sum(name in value for name in members) != 1:
            faults.append((None, f"answer exactly one of {', '.join(members)}"))
    return read, faults


def _checked_answer(spec: FieldSpec, value: Any, question: Mapping[str, Any]) -> Any:
    """One question's answer in the node's own form, held to the choices it offers; a pick of several picks each once."""
    read = read_answer(spec, value)
    offered = question.get("choices")
    if offered is not None:
        picked = _as_json(spec, read)
        picks = picked if spec.shape == "many" else [picked]
        ids = [choice["id"] for choice in offered]
        if any(item not in ids for item in picks):
            raise Refused("the answer is not one of the choices")
        if len({repr(item) for item in picks}) != len(picks):
            raise Refused("a choice is picked twice")
    return read


def read_answer(spec: FieldSpec, value: Any) -> Any:
    """``value`` read as one value of the output ``spec``: a table from records or a frame, anything else by its type."""
    if spec.shape == "table":
        return _read_table(spec, value)
    return _strictly(spec.adapter, value)


def _strictly(adapter: TypeAdapter[Any], value: Any) -> Any:
    """``value`` read by ``adapter`` from its JSON form, strictly: a yes or no is ``true`` or ``false``, a number is a
    number, never a text that spells one; a date is still its ISO text, as JSON writes it."""
    try:
        written = to_json(value, inf_nan_mode="constants")
    except PydanticSerializationError:
        raise Refused("the answer has no JSON form") from None
    try:
        return adapter.validate_json(written, strict=True)
    except ValidationError as error:
        raise Refused(value_free_errors(error)[0]["msg"]) from None


_RECORDS = TypeAdapter(list[dict[str, Any]])


def _read_table(spec: FieldSpec, value: Any) -> pl.DataFrame:
    """A table answer: a ``polars.DataFrame``, or a list of records; a table with a row type has exactly its columns."""
    if isinstance(value, pl.DataFrame):
        records = value.to_dicts()
    else:
        try:
            records = _RECORDS.validate_python(value)
        except ValidationError:
            raise Refused("a table is a list of records, one object per row") from None
    if spec.columns is None:
        try:
            return pl.DataFrame(records)
        except Exception:  # noqa: BLE001  polars' own account may quote a cell
            raise Refused("the records don't make a table") from None
    unknown = sorted({key for record in records for key in record} - spec.columns.keys())
    if unknown:
        raise Refused(f"the table has no column '{unknown[0]}'; it has {', '.join(spec.columns)}")
    columns = {}
    for column, adapter in spec.column_adapters.items():
        try:
            columns[column] = _strictly(adapter, [record.get(column) for record in records])
        except Refused as refused:
            raise Refused(f"column '{column}': {refused.reason}") from None
    return pl.DataFrame(columns, strict=False)


def _as_json(spec: FieldSpec, value: Any) -> Any:
    """A value of the output ``spec`` as JSON; a table as its records."""
    if spec.shape == "table":
        return to_jsonable_python(value.to_dicts())
    return spec.adapter.dump_python(value, mode="json")
