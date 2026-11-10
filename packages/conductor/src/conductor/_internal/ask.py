"""Asking a person: ``raise Ask(...)`` in a node, the questions a host shows, and the answer read back.

The node's return type is what the person answers, and it says how to ask:
``-> bool`` is a yes or no, ``-> Literal["low", "high"]`` one of those,
``-> str`` a text, ``-> int``, ``float`` or ``date`` a number or a date, a
frozen dataclass a form of one question per field, ``-> Table[R]`` a table
of ``R``'s columns, and ``-> Series[X]`` several of ``X``, so
``Series[Literal[...]]`` is a pick of several. ``Ask`` adds only what the run
knows: the prompt, the choices on offer, a proposed answer and a control.

A waiting row is a ``Waiting``, and each of its ``questions`` is the field
record ``describe()`` gives an output, plus what the ``Ask`` said: so a host
draws a question with the code that draws an input. An answer is checked
against that record and the node's types before anything runs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, get_args, get_origin

import polars as pl
from pydantic import TypeAdapter, ValidationError
from pydantic_core import to_jsonable_python

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import value_free_errors
from conductor._internal.fields import SKIPPED, FieldSpec, without_none

if TYPE_CHECKING:
    from conductor._internal.graph import PlacedNode
    from conductor._internal.node_definition import NodeDefinition


class _NotGiven:
    def __repr__(self) -> str:
        return "NOT_GIVEN"


_NOT_GIVEN: Any = _NotGiven()


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
        def assign(ticket: str, team: str) -> str:
            raise Ask(f"Who takes {ticket}?", choices=[Choice("ida", "Ida B."), Choice("bo", "Bo K.")])

    A ``BaseException``, so an ``except Exception`` in the node lets it through.
    """

    def __init__(
        self,
        prompt: str,
        *,
        choices: Sequence[Any] | Mapping[str, Sequence[Any]] | None = None,
        default: Any = _NOT_GIVEN,
        widget: Any = None,
        defaults: Mapping[str, Any] | None = None,
        widgets: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(prompt, str):
            raise TypeError(f"Ask's prompt is the text a person reads, not {type(prompt).__name__}")
        super().__init__(prompt)
        self.prompt, self.choices, self.default, self.widget = prompt, choices, default, widget
        self.defaults, self.widgets = defaults, widgets
        #: Whether the ``Ask`` adds anything to its questions, so a plain one costs nothing more.
        self._adds = choices is not None or default is not _NOT_GIVEN or widget is not None or bool(defaults) or bool(widgets)


class Waiting(ConductorModel):
    """A row of a node waiting for a person, as a host shows it.

    ``id`` is what ``run.answer`` takes: the node's id, then ``@`` and each
    place of the row, ``"approve@1"``. ``prompt`` is what the node's ``Ask``
    said; ``inputs`` is what the node received, as JSON. ``questions`` has one
    entry per output, the field record ``describe()`` gives it (``name``,
    ``title``, ``shape``, ``type``, ``schema``, ``required``, and
    ``description``, ``widget``, ``exclusive_group`` or ``columns`` when it
    has them), with ``choices`` (``[{"id", "title"}]``, from the ``Ask`` or a
    ``Literal``), ``default`` and ``widget`` as the ``Ask`` set them. A node
    with one output is answered with its value, one with several with a
    mapping by name.
    """

    id: str
    node_id: str
    row: tuple[int, ...]
    prompt: str
    questions: list[dict[str, Any]]
    inputs: dict[str, Any]


class WrongAsk(Exception):
    """A node raised ``Ask`` with something its outputs can't be asked with; the text says what, quoting no value."""


def asked(ask: Ask, definition: NodeDefinition[..., Any]) -> dict[str, dict[str, Any]]:
    """What ``ask`` adds to each question, by output name, as JSON; ``WrongAsk`` if it doesn't fit the node's outputs."""
    outputs = definition.outputs
    lone = next(iter(outputs)) if len(outputs) == 1 else None
    if lone is not None:
        if ask.defaults or ask.widgets or isinstance(ask.choices, Mapping):
            raise WrongAsk("the node has one output, so it takes default=, widget= and a list of choices")
        choices = {} if ask.choices is None else {lone: ask.choices}
        defaults = {} if ask.default is _NOT_GIVEN else {lone: ask.default}
        widgets = {} if ask.widget is None else {lone: ask.widget}
    else:
        if ask.default is not _NOT_GIVEN or ask.widget is not None or (ask.choices is not None and not isinstance(ask.choices, Mapping)):
            raise WrongAsk(f"the node has {len(outputs)} outputs, so it takes defaults=, widgets= and choices= by output name")
        choices, defaults, widgets = dict(ask.choices or {}), dict(ask.defaults or {}), dict(ask.widgets or {})
    added: dict[str, dict[str, Any]] = {}
    for kind, given in (("choices", choices), ("default", defaults), ("widget", widgets)):
        for name, value in given.items():
            spec = outputs.get(name)
            if spec is None:
                raise WrongAsk(f"the {kind} names '{name}', which is not an output; the outputs are {', '.join(outputs) or 'none'}")
            added.setdefault(name, {})[kind] = _added(spec, kind, value)
    for name, added_here in added.items():
        _check_default_among_choices(outputs[name], added_here)
    return added


def _added(spec: FieldSpec, kind: str, value: Any) -> Any:
    """One thing an ``Ask`` adds to a question, as JSON."""
    if kind == "widget":
        try:
            return to_jsonable_python(value)
        except Exception:  # noqa: BLE001  a value with no JSON form, such as a class passed bare
            raise WrongAsk(f"the widget for '{spec.name}' has no JSON form; give an instance, such as Textarea()") from None
    if kind == "default":
        if value is SKIPPED:
            raise WrongAsk(f"the default for '{spec.name}' is SKIPPED; a person answers a value")
        try:
            return _as_json(spec, read_value(spec, value))
        except _Refused as refused:
            raise WrongAsk(f"the default for '{spec.name}' is not what it returns: {refused.reason}") from None
    if spec.shape == "table":
        raise WrongAsk(f"'{spec.name}' is a table, which takes no choices")
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise WrongAsk(f"the choices for '{spec.name}' are a list")
    one = spec.item_adapter
    offered = []
    for choice in value:
        try:
            kept = one.dump_python(one.validate_python(choice.id if isinstance(choice, Choice) else choice), mode="json")
        except ValidationError as error:
            raise WrongAsk(f"a choice for '{spec.name}' is not what it returns: {value_free_errors(error)[0]['msg']}") from None
        offered.append({**choice.model_dump(mode="json"), "id": kept} if isinstance(choice, Choice) else {"id": kept, "title": str(kept)})
    return offered


def _check_default_among_choices(spec: FieldSpec, added: dict[str, Any]) -> None:
    if "default" not in added or "choices" not in added or added["default"] is None:
        return
    ids = [choice["id"] for choice in added["choices"]]
    proposed = added["default"] if spec.shape == "many" else [added["default"]]
    if any(item not in ids for item in proposed):
        raise WrongAsk(f"the default for '{spec.name}' is not one of its choices")


def questions(placed: PlacedNode, added: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """A waiting row's questions: each output's record, with what its ``Ask`` added."""
    out = []
    for name, question in placed.questions.items():
        extra = added.get(name)
        if extra:
            question = {**question, **extra}
            if "default" in extra:
                question["required"] = False
        else:
            question = dict(question)
        out.append(question)
    return out


def questions_of(definition: NodeDefinition[..., Any]) -> dict[str, dict[str, Any]]:
    """Each output's question before any ``Ask`` adds to it; a lone output takes the node's title."""
    lone = len(definition.outputs) == 1
    return {name: _question(spec, definition.title if lone else None) for name, spec in definition.outputs.items()}


def _question(spec: FieldSpec, node_title: str | None) -> dict[str, Any]:
    question = spec.describe()
    if node_title is not None and spec.param.title is None:
        question["title"] = node_title  # a lone output's own name, "result", says nothing to a person
    if not spec.required:  # a field of a returned dataclass with a default proposes it
        question["default"] = _as_json(spec, spec.default)
    question["required"] = spec.required and spec.param.exclusive_group is None and not (spec.shape == "one" and spec.optional)
    item = without_none(spec.type)
    if spec.shape != "table" and get_origin(item) is Literal:
        question["choices"] = [{"id": to_jsonable_python(v), "title": str(v)} for v in get_args(item)]
    return question


class _Refused(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def answer_outputs(placed: PlacedNode, value: Any, added: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], list[tuple[str | None, str]]]:
    """A person's answer as the node's outputs, or every fault in it as ``(question, reason)``; nothing of the value is quoted."""
    outputs = placed.definition.outputs
    asked_now = {name: {**question, **added.get(name, {})} for name, question in placed.questions.items()}
    if len(outputs) == 1:
        (name, spec), = outputs.items()
        try:
            return {name: _checked_answer(spec, value, asked_now[name])}, []
        except _Refused as refused:
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
                read[name] = read_value(spec, question["default"])
            elif not question["required"]:
                read[name] = None
            else:
                faults.append((name, f"'{name}' was not answered"))
        except _Refused as refused:
            faults.append((name, f"'{name}': {refused.reason}"))
        if group is not None:
            groups.setdefault(group, []).append(name)
    for members in groups.values():
        if sum(name in value for name in members) != 1:
            faults.append((None, f"answer exactly one of {', '.join(members)}"))
    return read, faults


def _checked_answer(spec: FieldSpec, value: Any, question: Mapping[str, Any]) -> Any:
    """One question's answer in the node's own form, held to the choices it offers; a pick of several picks each once."""
    read = read_value(spec, value)
    offered = question.get("choices")
    if offered is not None:
        picked = _as_json(spec, read)
        picks = picked if spec.shape == "many" else [picked]
        ids = [choice["id"] for choice in offered]
        if any(item not in ids for item in picks):
            raise _Refused("the answer is not one of the choices")
        if len({repr(item) for item in picks}) != len(picks):
            raise _Refused("a choice is picked twice")
    return read


def read_value(spec: FieldSpec, value: Any) -> Any:
    """``value`` read as one value of the output ``spec``: a table from records or a frame, anything else by its type."""
    if spec.shape == "table":
        return _read_table(spec, value)
    try:
        return spec.adapter.validate_python(value)
    except ValidationError as error:
        raise _Refused(value_free_errors(error)[0]["msg"]) from None


_RECORDS = TypeAdapter(list[dict[str, Any]])


def _read_table(spec: FieldSpec, value: Any) -> pl.DataFrame:
    """A table answer: a ``polars.DataFrame``, or a list of records; a table with a row type has exactly its columns."""
    if isinstance(value, pl.DataFrame):
        records = value.to_dicts()
    else:
        try:
            records = _RECORDS.validate_python(value)
        except ValidationError:
            raise _Refused("a table is a list of records, one object per row") from None
    if spec.columns is None:
        try:
            return pl.DataFrame(records)
        except Exception:  # noqa: BLE001  polars' own account may quote a cell
            raise _Refused("the records don't make a table") from None
    unknown = sorted({key for record in records for key in record} - spec.columns.keys())
    if unknown:
        raise _Refused(f"the table has no column '{unknown[0]}'; it has {', '.join(spec.columns)}")
    columns = {}
    for column, adapter in spec.column_adapters.items():
        try:
            columns[column] = adapter.validate_python([record.get(column) for record in records])
        except ValidationError as error:
            raise _Refused(f"column '{column}': {value_free_errors(error)[0]['msg']}") from None
    return pl.DataFrame(columns, strict=False)


def _as_json(spec: FieldSpec, value: Any) -> Any:
    """A value of the output ``spec`` as JSON; a table as its records."""
    if spec.shape == "table":
        return to_jsonable_python(value.to_dicts())
    return spec.adapter.dump_python(value, mode="json")
