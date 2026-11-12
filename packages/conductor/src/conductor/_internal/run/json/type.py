"""The shape of a saved run: the classes whose attributes are the keys of the JSON that ``run.to_json()`` writes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, get_args

from pydantic import ConfigDict, NonNegativeInt, PositiveInt
from pydantic import Field as PydanticField

from conductor._internal.ask import Question
from conductor._internal.run.state import EndedEarly, Going

#: Every class here refuses a key it does not declare, and writes ``inf`` and ``nan`` as text.
_SAVED = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

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
        questions: By call number, the question the call asked a person; the call is waiting unless it is done.
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
                  "fingerprints": ["e360f9745eb81eae40ab99d8", "190831fe7288be89bdb0cbd5", "831029ff10fb2182c5e370a2"],
                  "item_indexes": {"0": [0], "1": [1], "2": [2]}}

    ``outcomes`` has the shape of ``SavedOutcomes``: a list per output. Each outcome is written once, and its
    place in the lists is only a place: these are calls of the previous run, not of this one. So what
    identifies an outcome is written beside it, by that place: the fingerprint of the arguments its call got,
    and the item indexes the call had.

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

        {"graph": "4c169cfec3ad80cb", "state": {"status": "complete"}, "max_items": null,
         "inputs": {"lines.text": "a\nb\nc"},
         "outcomes": {"lines": {"states": ["done"], "outputs": {"result": [["a", "b", "c"]]}},
                      "shout": {"states": ["done", "done", "done"], "outputs": {"result": ["A!", "B!", "C!"]}},
                      "join": {"states": ["done"], "outputs": {"result": ["A!+B!+C!"]}}},
         "reusable": {},
         "code": {"lines": "351606d7bb07c2ef", "shout": "8a4f4574bb596ba6", "join": "16256b91d80c6610"},
         "complete": ["lines", "shout", "join"]}

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
        inputs: The graph inputs the caller gave, by address.
        outcomes: By node, the outcomes of each node that has any.
        reusable: By node, the outcomes of the previous run that this run may still reuse.
        code: By node, the code fingerprint of the node definition that its outcomes and its reusable outcomes
            come from, so that outcomes of other code are never reused.
        complete: The nodes that were complete when the run last stopped, so that going on does not report
            them complete again.
        tables_written_once: The outputs in ``outcomes`` written as ``null`` because another output has the
            same table; left out of the JSON when ``None``.
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
