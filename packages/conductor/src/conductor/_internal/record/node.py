"""``NodeRecord``: one node of a graph record, ``{"node": "shout", "inputs": {"mark": {"constant": "?"}}}``, and what
each of its inputs gets: ``RecordConstant``, ``RecordGraphInput`` or ``RecordEdges``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import MISSING, AfterValidator, ConfigDict, Discriminator, Field, JsonValue, Tag

from conductor._internal.base import Address, ConductorModel, Name


def _empty(value: Mapping[str, Any]) -> bool:
    return not value


#: A record holds JSON, which has no float that isn't finite: one is refused, so a record equals what its JSON reads back.
JSON_ONLY = ConfigDict(allow_inf_nan=False)


class RecordConstant(ConductorModel):
    """An input with the same value on every call in every run, ``{"constant": "?"}``: no caller can change it."""

    model_config = ConductorModel.model_config | JSON_ONLY

    constant: JsonValue


class RecordGraphInput(ConductorModel):
    """An input whose value the caller gives when a run starts, under the name of a graph input:
    ``{"graph_input": "text"}``, or with the value used when the caller gives none, ``{"graph_input": "mark",
    "default": "!"}``. Several inputs may name one graph input, and each then gets the caller's one value.

    Attributes:
        graph_input: The graph input's name: the keyword a caller gives the value by, ``graph.run(text="a b")``.
        default: The value when the caller gives none; left out, the input's own default, and the caller must
            give a value when the input has none.
    """

    model_config = ConductorModel.model_config | JSON_ONLY

    graph_input: Name
    default: JsonValue | MISSING = MISSING


def _sorted_once(sources: tuple[Address, ...]) -> tuple[Address, ...]:
    """Returns ``sources`` sorted, so two records that list one input's sources in another order are equal and
    save alike.

    Raises:
        ValueError: A source is listed twice.
    """
    if len(set(sources)) != len(sources):
        twice = next(str(source) for i, source in enumerate(sources) if source in sources[:i])
        raise ValueError(f"{twice} is listed twice")
    return tuple(sorted(sources, key=lambda source: (source.node, source.output, source.field or "")))


class RecordEdges(ConductorModel):
    """The edges into an input: the output each comes from, ``{"sources": [{"node": "lines", "output": "result"}]}``.

    Several edges into one input are alternatives, from opposite branches of one decision, so nothing depends on
    their order: they are kept sorted.
    """

    sources: Annotated[tuple[Address, ...], Field(min_length=1), AfterValidator(_sorted_once)]


def _entry_kind(entry: Any) -> str | None:
    """The kind of an input's entry, by the one key it has of ``"constant"``, ``"graph_input"`` and ``"sources"``: read
    from the JSON object, or from the class of an entry built in Python. ``None`` when the object has none of them,
    or more than one."""
    match entry:
        case Mapping():
            keys = entry.keys() & {"constant", "graph_input", "sources"}
            return next(iter(keys)) if len(keys) == 1 else None
        case RecordConstant():
            return "constant"
        case RecordGraphInput():
            return "graph_input"
        case RecordEdges():
            return "sources"
    return None


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
    #: What each input gets, by the input's name: exactly one of a constant, a graph input or the edges into it.
    inputs: Mapping[
        Name,
        Annotated[
            Annotated[RecordConstant, Tag("constant")] | Annotated[RecordGraphInput, Tag("graph_input")] | Annotated[RecordEdges, Tag("sources")],
            Discriminator(
                _entry_kind,
                custom_error_type="one_of_constant_graph_input_sources",
                custom_error_message="an input gets exactly one of a constant, a graph input or sources",
            ),
        ],
    ] = Field(default_factory=dict, exclude_if=_empty)
    #: The host's own: position, size, titles a person edited. Stored and never read here.
    display: Mapping[str, JsonValue] = Field(default_factory=dict, exclude_if=_empty)
