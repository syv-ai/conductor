"""``NodeRecord``: one node of a graph record, ``{"node": "shout", "inputs": {"mark": {"constant": "?"}}}``, and
``InputEntry``, what each of its inputs gets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Self

from pydantic import MISSING, AfterValidator, ConfigDict, Field, JsonValue, model_validator
from pydantic_core import PydanticCustomError

from conductor._internal.base import ConductorModel, Name, Source


def _empty(value: Mapping[str, Any]) -> bool:
    return not value


#: A record holds JSON, which has no float that isn't finite: one is refused, so a record equals what its JSON reads back.
JSON_ONLY = ConfigDict(allow_inf_nan=False)


def _sorted_once(sources: tuple[Source, ...]) -> tuple[Source, ...]:
    """Returns ``sources`` sorted, so two records that list one input's sources in another order are equal and
    save alike.

    Raises:
        PydanticCustomError: There is no source (``no_sources``), or one is listed twice (``repeated_source``).
    """
    if not sources:
        raise PydanticCustomError("no_sources", "an input's sources list at least one output")
    if len(set(sources)) != len(sources):
        twice = next(str(source) for i, source in enumerate(sources) if source in sources[:i])
        raise PydanticCustomError("repeated_source", "{source} is listed twice", {"source": twice})
    return tuple(sorted(sources, key=lambda source: (source.node, source.output, source.field or "")))


class InputEntry(ConductorModel):
    """What one input of a node gets in a graph record: exactly one of a constant, a graph input or sources.

    - ``{"constant": "?"}``: the same value on every call in every run; no caller can change it.
    - ``{"graph_input": "text"}``: the value the caller gives when a run starts, ``graph.run(text="a b")``. With
      ``"default"``, the value when the caller gives none; without, the caller must give one. Several inputs may name
      one graph input, and each then gets the caller's one value.
    - ``{"sources": [{"node": "lines", "output": "result"}]}``: the outputs the edges into the input come from.
      Several are alternatives, from opposite branches of one decision, so they are kept sorted.

    Attributes:
        constant: The constant.
        graph_input: The graph input's name.
        default: The graph input's default.
        sources: The outputs the edges come from.
    """

    model_config = ConductorModel.model_config | JSON_ONLY

    constant: JsonValue | MISSING = MISSING
    graph_input: Name | MISSING = MISSING
    default: JsonValue | MISSING = MISSING
    sources: Annotated[tuple[Source, ...], AfterValidator(_sorted_once)] | MISSING = MISSING

    @model_validator(mode="after")
    def _one_kind(self) -> Self:
        """Refuses an entry with more or less than one of a constant, a graph input and sources
        (``one_of_constant_graph_input_sources``), or a default without a graph input (``default_without_graph_input``)."""
        if [self.constant, self.graph_input, self.sources].count(MISSING) != 2:
            raise PydanticCustomError("one_of_constant_graph_input_sources", "an input gets exactly one of a constant, a graph input or sources")
        if self.default is not MISSING and self.graph_input is MISSING:
            raise PydanticCustomError("default_without_graph_input", "a default is the default of a graph input; give the input a graph_input")
        return self


class NodeRecord(ConductorModel):
    """One node of a graph record: which node definition it uses, and what each of its inputs gets.

    Its id is its key in ``GraphRecord.nodes``. The node's inputs and outputs are not stored here: compiling works them
    out (``graph.describe()``). An input the record leaves out gets the node's own default.
    """

    model_config = ConductorModel.model_config | JSON_ONLY

    #: The id of the node definition this node uses.
    node: str
    #: Left out of the saved record when it is 1, as is each field below when empty.
    version: int = Field(default=1, exclude_if=lambda version: version == 1)
    #: What each input gets, by the input's name.
    inputs: Mapping[Name, InputEntry] = Field(default_factory=dict, exclude_if=_empty)
    #: The host's own: position, size, titles a person edited. Stored and never read here.
    display: Mapping[str, JsonValue] = Field(default_factory=dict, exclude_if=_empty)
