"""The shape of a saved run: the classes whose attributes are the keys of the JSON that ``run.to_json()`` writes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, is_dataclass
from types import NoneType, UnionType
from typing import Annotated, Any, ClassVar, Literal, Union, get_args, get_origin, get_type_hints

from pydantic import ConfigDict, NonNegativeInt, PositiveInt
from pydantic import Field as PydanticField

from conductor._internal.ask import Question
from conductor._internal.base import NON_FINITE_FLOATS

from ..state import EndedEarly, Going

#: Every class here refuses a key it does not declare, and writes ``inf`` and ``nan`` as text.
_SAVED = ConfigDict(extra="forbid", ser_json_inf_nan=NON_FINITE_FLOATS)

#: What a saved run writes for one call: ``"done"`` when the call has outputs, ``"skipped"`` when it was skipped.
#: Any other call is written as ``null``.
SavedOutcomeState = Literal["done", "skipped"]
SAVED_DONE, SAVED_SKIPPED = get_args(SavedOutcomeState)


@dataclass(slots=True, kw_only=True)
class SavedOutcomes:
    """The outcomes of one node in a saved run: a list per output, with one entry per call.

    ``shout`` was called three times, for the lines ``a``, ``b`` and ``c``. It is saved as::

        "shout": {"states": ["done", "done", "done"], "outputs": {"result": ["A!", "B!", "C!"]}}

    In memory a run keeps one outcome per call. Only the JSON has a list per output, for three reasons. Pydantic
    validates a list in one call, where a record per call costs a call each. Each output's name is written
    once. And there is no key per call to get wrong: a call too many or too few is a list of the wrong length,
    which loading refuses.

    A call's place in the lists is its call number: the place of the call among its node's calls, in the order
    of their item indexes. A node inside two loops with calls for the item indexes ``(0, 0)``, ``(0, 1)`` and
    ``(1, 0)`` has the call numbers 0, 1 and 2. The item indexes are not written. Loading works them out from the
    lengths of the saved series.

    A call that is not done has ``null`` in every list, and the keys after ``outputs`` say why. Here ``shout``
    was not called for the second line, because its input ``line`` got ``None``::

        "shout": {"states": ["done", null, "done"], "outputs": {"result": ["A!", null, "C!"]},
                  "not_called_because": {"1": "line"}}

    A key after ``outputs`` that is ``None`` is left out of the JSON. The attributes of this class are the keys
    of the JSON, and nothing else declares them.

    Attributes:
        states: For each call, ``"done"`` when it has outputs (its function returned, or a person answered),
            ``"skipped"`` when it was skipped, else ``None``.
        outputs: By output name, the value of the output for each call; ``None`` for a call that has no outputs.
        questions: By call number, the question the call asked a person; the call is waiting unless it is done, or
            the saved run's ``answers`` has an answer for it.
        not_called_because: By call number, the input that got ``None`` and does not accept it, so that the
            call was not made.
        skipped: By output name, the call numbers of the done calls that returned ``SKIPPED`` for the output.
        skipped_items: By output name and call number, the indexes of the ``SKIPPED`` items in a returned series.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _SAVED
    states: list[SavedOutcomeState | None]
    outputs: dict[str, list[Any]]
    questions: dict[NonNegativeInt, Question] | None = None
    not_called_because: dict[NonNegativeInt, str] | None = None
    skipped: dict[str, list[NonNegativeInt]] | None = None
    skipped_items: dict[str, dict[NonNegativeInt, list[NonNegativeInt]]] | None = None


@dataclass(slots=True, kw_only=True)
class SavedReusable[OutcomesT]:
    r"""The outcomes of one node that a run may reuse from the previous run, in a saved run.

    After ``run.with_inputs(text="a\nb\nd")`` on the finished README run, the new run may reuse the three
    outcomes of ``shout``. Before it has called anything it saves them as::

        "shout": {"outcomes": {"states": ["done", "done", "done"], "outputs": {"result": ["A!", "B!", "C!"]}},
                  "fingerprints": ["<24 hex characters>", "<24 hex characters>", "<24 hex characters>"],
                  "item_indexes": {"0": [0], "1": [1], "2": [2]}}

    ``outcomes`` has the shape of ``SavedOutcomes``: a list per output. Each outcome is written once, and its
    place in the lists is only a place: these are calls of the previous run, not of this one. So what
    identifies an outcome is written beside it, by that place: the fingerprint of the arguments its call got,
    and the item indexes the call had. An argument fingerprint is a hash of the node's code fingerprint and the
    arguments as JSON.

    Attributes:
        outcomes: The outcomes: a ``SavedOutcomes`` in a saved run that is written, and the list of outcomes it
            describes in one that was read.
        fingerprints: For each outcome, the fingerprint of the arguments its call got.
        item_indexes: By place in the lists, the item indexes the call had in the previous run, for an outcome
            a function returned; left out of the JSON when ``None``.
        outer_item_fingerprints: By place in the lists, the fingerprint of the items of the outer loops the
            call was inside, for an answer or a waiting question; left out of the JSON when ``None``.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _SAVED
    outcomes: OutcomesT
    fingerprints: list[str]
    item_indexes: dict[NonNegativeInt, tuple[NonNegativeInt, ...]] | None = None
    outer_item_fingerprints: dict[NonNegativeInt, str] | None = None


@dataclass(slots=True, kw_only=True)
class SameAs:
    """One output value that a saved run does not write a second time, because another output has the same table.

    ``read`` returns a table and ``split`` returns that same table as a series of records. The saved run
    writes the table under ``read``, ``null`` under ``split``, and this entry under ``tables_written_once``::

        {"output": ["split", "result", 0], "same_as": ["read", "result", 0]}

    Loading gives both outputs the one table again.

    Attributes:
        output: The output written as ``null``: the node's name in the saved run, the output's name, and the
            call number.
        same_as: The output that is written in full, in the same three parts.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _SAVED
    output: tuple[str, str, NonNegativeInt]
    same_as: tuple[str, str, NonNegativeInt]


@dataclass(slots=True, kw_only=True)
class SavedRun[OutcomesT, ReusableT]:
    r"""A saved run: everything ``run.to_json()`` writes.

    The finished run of the README graph ``lines -> shout -> join`` on ``"a\nb\nc"`` is saved as::

        {"graph": "<16 hex characters>", "state": {"status": "complete"}, "max_items": null,
         "inputs": {"text": "a\nb\nc"},
         "outcomes": {"lines": {"states": ["done"], "outputs": {"result": [["a", "b", "c"]]}},
                      "shout": {"states": ["done", "done", "done"], "outputs": {"result": ["A!", "B!", "C!"]}},
                      "join": {"states": ["done"], "outputs": {"result": ["A!+B!+C!"]}}},
         "reusable": {},
         "code": {"lines": "<16 hex characters>", "shout": "<16 hex characters>", "join": "<16 hex characters>"},
         "complete": ["lines", "shout", "join"]}

    ``graph`` is the graph's fingerprint: a hash of the graph record and of the subgraphs it uses. Each value
    under ``code`` is a code fingerprint: a hash of the node's version, its function's code, its inputs and
    outputs, and the loops it is inside.

    A node is named by its node path joined by ``/``: ``shout`` in the graph itself, ``loud/shout`` inside the
    subgraph node ``loud``. A subgraph node is never called, so it has no key of its own.

    The class is used in both directions. Written, ``OutcomesT`` is ``SavedOutcomes`` and ``ReusableT`` is
    ``SavedReusable[SavedOutcomes]``. Read, the codec has already turned each of them into what a run keeps: a
    list of outcomes in call-number order, and a ``ReusableOutcomes``.

    Attributes:
        graph: The fingerprint of the run's graph; a saved run of another graph is refused by it before
            anything else is read.
        state: The run's ``state``: its status and, when it ended early, the problem that ended it.
        max_items: The run's item limit.
        inputs: The graph inputs the caller gave, by name.
        outcomes: By node, the outcomes of each node that has any.
        reusable: By node, the outcomes of the previous run that this run may still reuse.
        code: By node, the code fingerprint of the node definition that its outcomes and its reusable outcomes
            come from, so that outcomes of other code are never reused.
        complete: The nodes that were complete when the run last stopped, so that going on does not report
            them complete again.
        tables_written_once: The outputs in ``outcomes`` written as ``null`` because another output has the
            same table; left out of the JSON when ``None``.
        answers: By node and then call number, a person's answer to the node's ``Asked`` parameter, read by the
            parameter's type, for a call that waited and that the node is not yet called for: ``{"approve": {"1":
            true}}``. Its question is under the node's ``questions``. Left out of the JSON when ``None``.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _SAVED
    graph: str
    state: Annotated[Going | EndedEarly, PydanticField(discriminator="status")]
    max_items: PositiveInt | None
    inputs: dict[str, Any]
    outcomes: dict[str, OutcomesT]
    reusable: dict[str, ReusableT]
    code: dict[str, str]
    complete: list[str]
    tables_written_once: list[SameAs] | None = None
    answers: dict[str, dict[NonNegativeInt, Any]] | None = None


#: A saved run as the codec reads it, with each node's outcomes and reusable outcomes in their saved form.
_AS_READ = SavedRun[SavedOutcomes, SavedReusable[SavedOutcomes]]


def steps_to_value(loc: Sequence[str | int]) -> list[str | int]:
    """Returns the steps of ``loc``, a place pydantic names in a saved run, up to the value found there, and none
    inside the value, since a step inside it may be a key of the value's own dict.

    ``("outcomes", "g", "outputs", "result", 0, "counts", "ida")`` gives ``["outcomes", "g", "outputs", "result",
    0]``: the value of ``g.result`` on call 0 is ``{"counts": {"ida": ...}}``. ``("inputs", "text")`` and
    ``("outcomes", "g", "states", 0)`` are kept whole. The steps are walked through the classes above: a value is
    wherever they declare a type other than one of them, a dict or a list, such as ``Any``.
    """
    kept: list[str | int] = []
    hint: Any = _AS_READ
    bound: dict[Any, Any] = {}
    for step in loc:
        hint = _inside(hint, step, bound)
        if hint is None:
            break
        kept.append(step)
    return kept


def _inside(hint: Any, step: str | int, bound: dict[Any, Any]) -> Any:
    """Returns the hint that ``step`` leads to inside ``hint``, one of the hints of a saved run's classes; ``None``
    when ``hint`` is a value's. A step the class does not declare leads to ``Any``. ``bound`` gathers what each
    class's type parameters are, as ``SavedRun[SavedOutcomes, ...]`` says them."""
    hint = bound.get(hint, hint)
    members = [member for member in get_args(hint) if member is not NoneType] if get_origin(hint) in (Union, UnionType) else [hint]
    if len(members) > 1:  # one of several classes, as ``state`` is: pydantic's step is the tag of the class it read
        fields = [get_type_hints(member) for member in members if is_dataclass(member)]
        if len(fields) < len(members):
            return None  # a union with a type that is not a saved run's class: a value's
        return next((declared[str(step)] for declared in fields if str(step) in declared), hint)
    hint = members[0]  # an optional key: what it is when it is there
    origin, args = get_origin(hint) or hint, get_args(hint)
    if isinstance(origin, type) and is_dataclass(origin):
        bound.update(zip(origin.__type_params__, args, strict=False))
        return get_type_hints(origin).get(str(step), Any)
    if origin in (dict, Mapping):
        return args[1]
    if origin is list:
        return args[0]
    return None
