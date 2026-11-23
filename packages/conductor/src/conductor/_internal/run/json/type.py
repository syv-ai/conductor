"""The shape of a saved run: the JSON that ``run.to_json()`` writes, as pydantic models.

Each class says what holds for every graph's saved runs. ``codec.py`` makes a subclass of each for one graph,
which narrows the fields to that graph's own node names and value types.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Literal, Self, get_args

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, model_validator
from pydantic import Field as PydanticField

from conductor._internal.base import JSON_CONFIG
from conductor._internal.problem import conductor_error
from conductor._internal.series import SeriesItems

from ..outcome import Unanswered
from ..state import EndedEarly

#: The config of every class here. A key a class doesn't declare is refused, ``inf`` and ``nan`` are written as
#: text, and ``bytes`` as base64 (``JSON_CONFIG``). A saved run is read strictly, as a value that enters a graph is,
#: so ``"Yes"`` is never a ``bool`` and ``"3"`` never an ``int`` (the codec's ``_read_strictly``). The config doesn't leave defaults out: ``run_to_json`` writes with ``exclude_defaults=True``, so a key
#: whose value is its default, an empty dict or list, is left out of the JSON.
#:
#: The classes are plain pydantic models, not ``ConductorModel``s. The codec subclasses them with the graph's own
#: value types, and ``ConductorModel`` refuses a class with a field that has no JSON form as soon as the class is
#: built. A node may return a host's type that has no JSON form: its run runs, and only saving it is refused, with
#: ``run_unwritable`` naming the output.
_SAVED = ConfigDict(**JSON_CONFIG, extra="forbid", strict=True)

#: What a saved run writes for one call's state. ``"done"`` means it has its outputs, because its function
#: returned or a person answered. ``"skipped"`` means it was skipped. Any other call is written as ``null``.
SavedOutcomeState = Literal["done", "skipped"]
SAVED_DONE, SAVED_SKIPPED = get_args(SavedOutcomeState)


class SavedOutcomes(BaseModel):
    """The outcomes of one node in a saved run: one list per output, with an entry per call.

    ``shout`` was called three times, for the lines ``a``, ``b`` and ``c``. It is saved as::

        "shout": {"states": ["done", "done", "done"], "outputs": {"result": ["A!", "B!", "C!"]},
                  "fingerprints": ["<fp a>", "<fp b>", "<fp c>"]}

    A call fingerprint (``<fp a>``) is a 24-character hash of the node's code fingerprint and the arguments as JSON,
    and for a node that may ask, of the call's item in every loop it was inside too. It is taken when the call is
    made, so it keeps the code of that time: a later run reuses the outcome only if its call has the same code.

    In memory, a run stores one outcome per call. Only the JSON has a list per output, for three reasons. Pydantic
    checks a whole list in one go, where a model per call would cost a check each. Each output's name is written
    once. And there is no key per call to get wrong: a call too many or too few makes a list of the wrong length,
    which reading refuses.

    A call's place in the lists is its call number. That is its place among the node's calls, in the order of their
    item indexes. A node inside two loops with calls at ``(0, 0)``, ``(0, 1)`` and ``(1, 0)`` has the call numbers
    0, 1 and 2. The item indexes aren't written. Reading works them out from the lengths of the saved series.

    A call that isn't done has ``null`` in every list, and the keys after ``outputs`` say why. Here ``shout``
    wasn't called for the second line, because its input ``line`` got ``None``::

        "shout": {"states": ["done", null, "done"], "outputs": {"result": ["A!", null, "C!"]},
                  "fingerprints": ["<fp a>", null, "<fp c>"], "none_inputs": {"1": "line"}}

    A call of a node that asked keeps what its ``Ask`` said under ``questions``, while it waits and once a person
    has answered. Here ``approve`` asked about the second line, and the person said no::

        "approve": {"states": ["done", "done", "done"], "outputs": {"result": [true, false, true]},
                    "fingerprints": ["<fp a>", "<fp b>", "<fp c>"],
                    "questions": {"1": {"prompt": "Approve 'b'?", "defaults": {"result": true}}}}

    A key after ``outputs`` is left out of the JSON when it is empty.

    The lists hold together as a run writes them, or the saved run is refused with ``saved_lists_disagree`` at the
    node (``_lists_agree``). Each output's list and ``fingerprints`` are as long as ``states``. ``skipped``,
    ``skipped_items``, ``questions`` and ``none_inputs`` name only calls that can have them: ``skipped_items`` an
    item of a ``Series`` saved as a list, ``questions`` a call that is done or waiting, and ``none_inputs`` a call
    with no state and no question. Only a done or a waiting call has a fingerprint. A question names only the node's
    outputs (``question_disagreement``). And a done call has ``null`` where its output's type, or its items', takes
    no ``None`` only where ``skipped`` or ``skipped_items`` says the node returned ``SKIPPED`` there
    (``takes_none``): a node that returned ``None`` there would have failed with ``invalid_output``.

    Attributes:
        states: For each call: ``"done"`` when its function returned or a person answered, ``"skipped"`` when it was
            skipped, and ``None`` otherwise.
        outputs: By output name, the output's value for each call. ``None`` for a call with no outputs.
        fingerprints: For each call, its call fingerprint: for a done call and a waiting one. ``None`` for any
            other call, and for one whose arguments or items have no JSON form.
        questions: By call number, what the ``Ask`` of a call that asked said: its prompt, choices and defaults. A
            waiting call has ``None`` for its state. An answered one is ``"done"``, and keeps what the person was
            shown beside their answer under ``outputs``.
        none_inputs: By call number, the input that got ``None`` and doesn't take it, so the call wasn't
            made.
        skipped: By output name, the call numbers of the done calls that returned ``SKIPPED`` for that output.
        skipped_items: By output name and call number, the indexes of the ``SKIPPED`` items in a returned series.
        takes_none: In a graph's codec, by output name, whether the output's type takes ``None``, and whether its
            items do: ``(False, True)`` for ``Series[str | None]``.
    """

    model_config = _SAVED
    states: list[SavedOutcomeState | None]
    outputs: dict[str, list[Any]]
    fingerprints: list[str | None]
    questions: dict[NonNegativeInt, Unanswered] = PydanticField(default_factory=dict)
    none_inputs: dict[NonNegativeInt, str] = PydanticField(default_factory=dict)
    skipped: dict[str, list[NonNegativeInt]] = PydanticField(default_factory=dict)
    skipped_items: dict[str, dict[NonNegativeInt, list[NonNegativeInt]]] = PydanticField(
        default_factory=dict
    )
    takes_none: ClassVar[Mapping[str, tuple[bool, bool]]]

    @model_validator(mode="after")
    def _lists_agree(self) -> Self:
        """Refuses lists no run writes, as the class says.

        Raises:
            PydanticCustomError: ``saved_lists_disagree``, since only a saved run can give one: its message names
                outputs, call numbers and keys, and no value.
        """
        if (wrong := self._disagreement()) is not None:
            raise conductor_error("saved_lists_disagree", wrong)
        return self

    def _disagreement(self) -> str | None:
        """Returns what in the lists disagrees, the first fault in the order the class says them, or ``None``."""
        calls = len(self.states)
        if any(len(values) != calls for values in self.outputs.values()):
            return "the list of an output is not as long as 'states'"
        if len(self.fingerprints) != calls:
            return "'fingerprints' is not as long as 'states'"
        if not self._calls_can_have_them():
            return (
                "'skipped', 'skipped_items', 'questions' or 'none_inputs' names a call that does not exist or "
                "cannot have it"
            )
        for question in self.questions.values():
            if (wrong := question_disagreement(question, self.outputs)) is not None:
                return wrong
        if any(
            fingerprint is not None
            and self.states[call_number] != SAVED_DONE
            and call_number not in self.questions
            for call_number, fingerprint in enumerate(self.fingerprints)
        ):
            return "a call that is neither done nor waiting has a fingerprint"
        return self._null_where_none_is_refused()

    def _calls_can_have_them(self) -> bool:
        """Says whether each call number under ``skipped``, ``skipped_items``, ``questions`` and ``none_inputs`` is
        a call that can have that entry, as the class says."""
        calls = len(self.states)
        if any(n >= calls for numbers in self.skipped.values() for n in numbers):
            return False
        for name, by_call_number in self.skipped_items.items():
            skipped = set(self.skipped.get(name, ()))
            for call_number, indexes in by_call_number.items():
                # only a Series saved as a list, and not skipped as a whole, has its items listed
                items = self.outputs[name][call_number] if call_number < calls else None
                if call_number in skipped or not isinstance(items, SeriesItems):
                    return False
                if any(index >= len(items) for index in indexes):
                    return False
        if any(
            n >= calls or self.states[n] == SAVED_SKIPPED for n in self.questions
        ):  # a question is kept by a call that is done or waiting
            return False
        return not any(
            n >= calls or self.states[n] is not None or n in self.questions
            for n in self.none_inputs
        )

    def _null_where_none_is_refused(self) -> str | None:
        """Returns where a done call has ``null`` that the output's type, or its items', takes as no ``None``.

        A saved run writes ``SKIPPED`` as ``null``, so the codec takes ``null`` for every output and every item of
        a ``Series``. A ``null`` that ``skipped`` or ``skipped_items`` doesn't list is a ``None`` the node returned.
        Say ``split`` returns ``Series[str]``, and its saved ``["a", null]`` lists no skipped item: it is refused.
        """
        for name, values in self.outputs.items():
            takes_none, items_take_none = self.takes_none[name]
            # an output whose value and items may be None has nothing to refuse
            if takes_none and items_take_none:
                continue
            skipped = set(self.skipped.get(name, ()))
            skipped_items = self.skipped_items.get(name, {})
            for call_number, value in enumerate(values):
                # most values are not None, so a call's state is looked up only for a value that may be refused
                if (
                    value is None
                    and not takes_none
                    and self.states[call_number] == SAVED_DONE
                    and call_number not in skipped
                ):
                    return f"call {call_number} is done and has null for '{name}', whose type takes no None"
                if (
                    not items_take_none
                    and type(value) is SeriesItems
                    and self.states[call_number] == SAVED_DONE
                    and any(item is None for item in value)
                ):
                    listed = set(skipped_items.get(call_number, ()))
                    if any(
                        item is None and index not in listed for index, item in enumerate(value)
                    ):
                        return f"call {call_number} is done and has null for an item of '{name}', whose items take no None"
        return None


class SavedReusable(SavedOutcomes):
    r"""The outcomes of one node that a run may still reuse from a previous run, in a saved run.

    Say the README run has finished, and ``run.with_inputs({"text": "a\nb\nd"})`` makes a new run. The new run may reuse
    the three outcomes of ``shout``. Before it has called anything, it saves them as::

        "shout": {"states": ["done", "done", "done"], "outputs": {"result": ["A!", "B!", "C!"]},
                  "fingerprints": ["<fp a>", "<fp b>", "<fp c>"], "item_indexes": [[0], [1], [2]]}

    It has the keys of ``SavedOutcomes``, a list per output. But here a place in the lists isn't a call number of this
    run: these are calls of the previous run. So the item indexes each call had are written beside them, in a list of
    their own.

    Each outcome is written once, and every node's the same way, whether it may ask or not. Only the order matters
    differently. For a node that never asks, the first listed with a fingerprint is the one any call with that
    fingerprint reuses, and each other one is reused only by the call with its item indexes. A node that may ask
    has its outcomes with one fingerprint taken by its calls in the order listed. It never reuses an outcome by its
    item indexes, so its ``item_indexes`` are read only to be written again when the run is saved.

    Only a call that was made can be reused, so every outcome here is a done or a waiting call with a fingerprint.
    A ``null`` fingerprint is refused, and so is a skipped call, or one not made, which has a ``null`` one.

    Attributes:
        fingerprints: For each outcome, its call fingerprint.
        item_indexes: For each outcome, the item indexes its call had in the previous run.
    """

    fingerprints: list[str]  # type: ignore[assignment]  # narrower than SavedOutcomes': no outcome here lacks one
    item_indexes: list[tuple[NonNegativeInt, ...]]

    @model_validator(mode="after")
    def _one_call_each(self) -> Self:
        """Refuses ``item_indexes`` that aren't one per outcome, once the lists agree (``_lists_agree``).

        Raises:
            PydanticCustomError: ``saved_lists_disagree``.
        """
        if len(self.item_indexes) != len(self.states):
            raise conductor_error(
                "saved_lists_disagree", "'item_indexes' is not as long as 'states'"
            )
        return self


class SavedRun(BaseModel):
    r"""A saved run: everything ``run.to_json()`` writes.

    The finished README run, ``lines -> shout -> join`` on ``"a\nb\nc"``, is saved as::

        {"graph": "<16 hex characters>", "ended_early": null, "max_items": null,
         "inputs": {"text": "a\nb\nc"},
         "outcomes": {"lines": {"states": ["done"], "outputs": {"result": [["a", "b", "c"]]}, "fingerprints": ["<fp>"]},
                      "shout": {"states": ["done", "done", "done"], "outputs": {"result": ["A!", "B!", "C!"]},
                                "fingerprints": ["<fp a>", "<fp b>", "<fp c>"]},
                      "join": {"states": ["done"], "outputs": {"result": ["A!+B!+C!"]}, "fingerprints": ["<fp>"]}},
         "reusable": {},
         "complete": ["lines", "shout", "join"]}

    ``graph`` is the graph's fingerprint, a hash of the graph record and of the subgraphs it uses.

    A node is named by its node path joined with ``/``: ``shout`` in the graph itself, ``loud/shout`` inside the
    subgraph node ``loud``. A subgraph node is never called, so it has no key of its own.

    Attributes:
        graph: The fingerprint of the run's graph. A saved run of another graph is refused by it.
        ended_early: How the last ``resume()`` ended the run early: its status and the problem that ended it.
            ``null`` when it didn't. A resume still going is never saved: a copy made during one is saved with
            ``null`` here, and when read back its status comes from its outcomes. A declined run keeps the question
            declined here too, under ``declined``.
        max_items: The run's item limit.
        inputs: The graph inputs the caller gave, by name.
        outcomes: By node, the outcomes of each node that has any.
        reusable: By node, the outcomes of the previous run that this run may still reuse.
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
    complete: list[str]
    outcomes_classes: ClassVar[dict[str, type[SavedOutcomes]]]
    reusable_classes: ClassVar[dict[str, type[SavedReusable]]]


def question_disagreement(question: Unanswered, outputs: Iterable[str]) -> str | None:
    """Returns why a saved question isn't one an ``Ask`` of a node with these outputs gives, or ``None`` if it is.

    An ``Ask`` gives choices and defaults only for the node's outputs. A saved question is a record of what a person
    was shown, so its values are kept as they were saved, and never read again by the output's type: a deploy may
    have changed the type, and the person was shown them all the same. The codec checks the rest of its shape: a
    prompt and titles as text, choices and defaults by output name, at least one choice and none twice
    (``Choices``), values as JSON. An output the node doesn't have is what no deploy explains.
    """
    if not {*question.choices, *question.defaults} <= set(outputs):
        return "a question names an output the node does not have"
    return None
