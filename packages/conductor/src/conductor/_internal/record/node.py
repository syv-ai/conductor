"""``NodeRecord``, one node of a graph record, and ``InputEntry``, what one of its inputs gets.

In JSON, a node looks like ``{"node": "shout", "inputs": {"mark": {"constant": "?"}}}``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Self

from pydantic import MISSING, AfterValidator, ConfigDict, Field, JsonValue, model_validator
from pydantic_core import PydanticCustomError

from conductor._internal.base import ConductorModel, Name, Source


def _empty(value: Mapping[str, Any]) -> bool:
    """Whether a mapping is empty, so it is left out when the record is saved."""
    return not value


#: A record is JSON, and JSON has no ``inf`` or ``nan``. Refusing them keeps a record equal to its JSON read back.
JSON_ONLY = ConfigDict(allow_inf_nan=False)


def _sorted_once(sources: tuple[Source, ...]) -> tuple[Source, ...]:
    """Checks an input's sources and returns them sorted.

    Sorting means two records that list the same sources in a different order are equal, and save the same.

    Raises:
        PydanticCustomError: ``no_sources`` if the list is empty, ``repeated_source`` if a source is listed twice.
    """
    if not sources:
        raise PydanticCustomError("no_sources", "an input's sources list at least one output")
    seen: set[Source] = set()
    for source in sources:
        if source in seen:
            raise PydanticCustomError(
                "repeated_source", "{source} is listed twice", {"source": str(source)}
            )
        seen.add(source)
    return tuple(
        sorted(sources, key=lambda source: (source.node, source.output, source.field or ""))
    )


class InputEntry(ConductorModel):
    """What one input of a node gets in a graph record: a constant, a graph input or sources.

    Each entry is exactly one of the three:

    - ``{"constant": "?"}`` is a fixed value. The input has it on every call of every run, and no caller can change
      it.
    - ``{"graph_input": "text"}`` is a value the caller gives when a run starts, as in ``graph.run(text="a b")``.
      Add ``"default"`` for a value to use when the caller gives none. Without one, the caller must give it.
      Several inputs may name the same graph input, and then they all get the caller's one value.
    - ``{"sources": [{"node": "lines", "output": "result"}]}`` lists the outputs whose edges lead into the input.
      A source may also name one field of a record, as in ``{"node": "read", "output": "result", "field":
      "email"}``. Several sources are alternatives, from opposite branches of one decision.

    Attributes:
        constant: The fixed value.
        graph_input: The name of the graph input.
        default: The graph input's default.
        sources: The outputs the edges come from, sorted.
    """

    model_config = ConductorModel.model_config | JSON_ONLY

    constant: JsonValue | MISSING = MISSING
    graph_input: Name | MISSING = MISSING
    default: JsonValue | MISSING = MISSING
    sources: Annotated[tuple[Source, ...], AfterValidator(_sorted_once)] | MISSING = MISSING

    @model_validator(mode="after")
    def _one_kind(self) -> Self:
        """Checks that the entry is exactly one kind, and has a default only beside a graph input.

        Raises:
            PydanticCustomError: ``one_of_constant_graph_input_sources`` if the entry is none of the three kinds or
                more than one. ``default_without_graph_input`` if it has a default but no graph input.
        """
        if [self.constant, self.graph_input, self.sources].count(MISSING) != 2:
            raise PydanticCustomError(
                "one_of_constant_graph_input_sources",
                "an input gets exactly one of a constant, a graph input or sources",
            )
        if self.default is not MISSING and self.graph_input is MISSING:
            raise PydanticCustomError(
                "default_without_graph_input",
                "a default is the default of a graph input; give the input a graph_input",
            )
        return self


class NodeRecord(ConductorModel):
    """One node of a graph record: which node it uses, and what each of its inputs gets.

    The node's id is its key in ``GraphRecord.nodes``. The record doesn't list the node's inputs and outputs.
    Compiling the graph works those out, and ``graph.describe()`` shows them. An input the record leaves out gets
    the node's own default.
    """

    model_config = ConductorModel.model_config | JSON_ONLY

    #: The id of the node definition this node uses.
    node: str
    #: The version of the node definition. Left out of the saved record when it is 1, like each field below when empty.
    version: int = Field(default=1, exclude_if=lambda version: version == 1)
    #: What each input gets, by the input's name.
    inputs: Mapping[Name, InputEntry] = Field(default_factory=dict, exclude_if=_empty)
    #: The host's own data, like position, size or a title a person edited. Conductor saves it and never looks at it.
    display: Mapping[str, JsonValue] = Field(default_factory=dict, exclude_if=_empty)
