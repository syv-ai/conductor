"""The shape of a saved run: the JSON that ``run.to_json()`` writes, as pydantic models.

Each class says what holds for every graph's saved runs. ``codec.py`` makes a subclass of each for one graph,
which narrows the fields to that graph's own node names and value types.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal, get_args

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt
from pydantic import Field as PydanticField

from conductor._internal.base import NON_FINITE_FLOATS

from ..outcome import Unanswered
from ..state import EndedEarly

#: The config of every class here. A key a class doesn't declare is refused, and ``inf`` and ``nan`` are written
#: as text. The config doesn't leave defaults out: ``run_to_json`` writes with ``exclude_defaults=True``, so a key
#: whose value is its default, an empty dict or list, is left out of the JSON.
#:
#: The classes are plain pydantic models, not ``ConductorModel``s. The codec subclasses them with the graph's own
#: value types, and ``ConductorModel`` refuses a class with a field that has no JSON form as soon as the class is
#: built. A node may return a host's type that has no JSON form: its run runs, and only saving it is refused, with
#: ``run_unwritable`` naming the output.
_SAVED = ConfigDict(extra="forbid", ser_json_inf_nan=NON_FINITE_FLOATS)

#: What a saved run writes for one call's state. ``"done"`` means it has its outputs, because its function
#: returned or a person answered. ``"skipped"`` means it was skipped. Any other call is written as ``null``.
SavedOutcomeState = Literal["done", "skipped"]
SAVED_DONE, SAVED_SKIPPED = get_args(SavedOutcomeState)


class SavedOutcomes(BaseModel):
    """The outcomes of one node in a saved run: one list per output, with an entry per call.

    ``shout`` was called three times, for the lines ``a``, ``b`` and ``c``. It is saved as::

        "shout": {"states": ["done", "done", "done"], "outputs": {"result": ["A!", "B!", "C!"]}}

    In memory, a run stores one outcome per call. Only the JSON has a list per output, for three reasons. Pydantic
    checks a whole list in one go, where a record per call would cost a check each. Each output's name is written
    once. And there is no key per call to get wrong: a call too many or too few makes a list of the wrong length,
    which reading refuses.

    A call's place in the lists is its call number. That is its place among the node's calls, in the order of their
    item indexes. A node inside two loops with calls at ``(0, 0)``, ``(0, 1)`` and ``(1, 0)`` has the call numbers
    0, 1 and 2. The item indexes aren't written. Reading works them out from the lengths of the saved series.

    A call that isn't done has ``null`` in every list, and the keys after ``outputs`` say why. Here ``shout``
    wasn't called for the second line, because its input ``line`` got ``None``::

        "shout": {"states": ["done", null, "done"], "outputs": {"result": ["A!", null, "C!"]},
                  "none_inputs": {"1": "line"}}

    A key after ``outputs`` is left out of the JSON when it is empty.

    Attributes:
        states: For each call: ``"done"`` when its function returned or a person answered, ``"skipped"`` when it was
            skipped, and ``None`` otherwise.
        outputs: By output name, the output's value for each call. ``None`` for a call with no outputs.
        questions: By call number, what the ``Ask`` of a waiting call said: its prompt, choices and defaults.
        none_inputs: By call number, the input that got ``None`` and doesn't take it, so the call wasn't
            made.
        skipped: By output name, the call numbers of the done calls that returned ``SKIPPED`` for that output.
        skipped_items: By output name and call number, the indexes of the ``SKIPPED`` items in a returned series.
    """

    model_config = _SAVED
    states: list[SavedOutcomeState | None]
    outputs: dict[str, list[Any]]
    questions: dict[NonNegativeInt, Unanswered] = PydanticField(default_factory=dict)
    none_inputs: dict[NonNegativeInt, str] = PydanticField(default_factory=dict)
    skipped: dict[str, list[NonNegativeInt]] = PydanticField(default_factory=dict)
    skipped_items: dict[str, dict[NonNegativeInt, list[NonNegativeInt]]] = PydanticField(
        default_factory=dict
    )


class SavedReusable(BaseModel):
    r"""The outcomes of one node that a run may still reuse from a previous run, in a saved run.

    Say the README run has finished, and ``run.with_inputs({"text": "a\nb\nd"})`` makes a new run. The new run may reuse
    the three outcomes of ``shout``. Before it has called anything, it saves them as::

        "shout": {"outcomes": {"states": ["done", "done", "done"], "outputs": {"result": ["A!", "B!", "C!"]}},
                  "fingerprints": ["<fp a>", "<fp b>", "<fp c>"],
                  "item_indexes": {"0": [0], "1": [1], "2": [2]}}

    ``outcomes`` has the shape of ``SavedOutcomes``, a list per output. But here a place in the lists isn't a call
    number of this run: these are calls of the previous run. So what tells the outcomes apart is written beside them,
    by place: each call's fingerprint, and the item indexes it had. A call fingerprint (``<fp a>``) is a 24-character hash of the node's code
    fingerprint and the arguments as JSON, and for a node that may ask, of the call's item in every loop it was
    inside too.

    Each outcome is written once. Of the outcomes with one fingerprint, the first listed is the one any call with
    that fingerprint reuses; each other one is reused only by the call with its item indexes. A node that may ask
    has no item indexes: its calls with one fingerprint take its outcomes in the order they are listed.

    Attributes:
        outcomes: The outcomes, as the node's subclass of ``SavedOutcomes`` in the graph's codec.
        fingerprints: For each outcome, its call's fingerprint.
        item_indexes: By place in the lists, the item indexes the call had in the previous run. Every outcome of a
            node that never asks has them, and no outcome of a node that may ask.
    """

    model_config = _SAVED
    outcomes: SavedOutcomes
    fingerprints: list[str]
    item_indexes: dict[NonNegativeInt, tuple[NonNegativeInt, ...]] = PydanticField(
        default_factory=dict
    )


class SavedRun(BaseModel):
    r"""A saved run: everything ``run.to_json()`` writes.

    The finished README run, ``lines -> shout -> join`` on ``"a\nb\nc"``, is saved as::

        {"graph": "<16 hex characters>", "ended_early": null, "max_items": null,
         "inputs": {"text": "a\nb\nc"},
         "outcomes": {"lines": {"states": ["done"], "outputs": {"result": [["a", "b", "c"]]}},
                      "shout": {"states": ["done", "done", "done"], "outputs": {"result": ["A!", "B!", "C!"]}},
                      "join": {"states": ["done"], "outputs": {"result": ["A!+B!+C!"]}}},
         "reusable": {},
         "code": {"lines": "<16 hex characters>", "shout": "<16 hex characters>", "join": "<16 hex characters>"},
         "complete": ["lines", "shout", "join"]}

    ``graph`` is the graph's fingerprint, a hash of the graph record and of the subgraphs it uses. Each value under
    ``code`` is a node's code fingerprint, a hash of its ``@node`` id and version, its function's code, its inputs
    and outputs, the loops it is inside, and whether it may ask.

    A node is named by its node path joined with ``/``: ``shout`` in the graph itself, ``loud/shout`` inside the
    subgraph node ``loud``. A subgraph node is never called, so it has no key of its own.

    Attributes:
        graph: The fingerprint of the run's graph. A saved run of another graph is refused by it.
        ended_early: How the last ``resume()`` ended the run early: its status and the problem that ended it.
            ``null`` when it didn't. A resume still going is never saved: a copy made during one is saved with
            ``null`` here, and when read back its status comes from its outcomes.
        max_items: The run's item limit.
        inputs: The graph inputs the caller gave, by name.
        outcomes: By node, the outcomes of each node that has any.
        reusable: By node, the outcomes of the previous run that this run may still reuse.
        code: By node, the code fingerprint of the node definition its outcomes and reusable outcomes came from.
            That way, outcomes of other code are never reused.
        complete: The nodes that were complete when the run last stopped, so going on doesn't report them complete
            again.
        outcomes_classes: In a graph's codec, the class of each node's ``outcomes``, by node.
        reusable_classes: In a graph's codec, the class of each node's ``reusable`` entry, by node.
    """

    model_config = _SAVED
    graph: str
    ended_early: EndedEarly | None
    max_items: PositiveInt | None
    inputs: dict[str, Any]
    outcomes: dict[str, SavedOutcomes]
    reusable: dict[str, SavedReusable]
    code: dict[str, str]
    complete: list[str]
    outcomes_classes: ClassVar[dict[str, type[SavedOutcomes]]]
    reusable_classes: ClassVar[dict[str, type[SavedReusable]]]
