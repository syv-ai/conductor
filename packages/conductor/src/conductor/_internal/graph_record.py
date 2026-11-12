"""``GraphRecord`` and ``NodeRecord``: a graph as it is stored.

A graph record is its nodes and its edges, and nothing else::

    {
        "nodes": [
            {"id": "read", "node": "read_file", "constants": {"path": "in.txt"}},
            {"id": "up", "node": "upper"}
        ],
        "edges": [["read.text", "up.text"]]
    }

A node names the definition it uses (``node``) and the version (``version``,
1 when left out). ``constants`` are typed into its inputs and fixed there: no
caller may change them. ``defaults`` fill an input that stays open, so a
caller may still give it another value. A list given to an input that takes
one value, by the caller or typed in, is many values: the node runs once per
item. ``display`` is the host's own: conductor stores it and never reads it. An empty field is left out when the
record is saved, so a bare node is ``{"id": "up", "node": "upper"}``.

Edges are a list of ``[from, to]`` addresses, ``node.field`` or
``node.output.field`` for one field of a record, kept sorted, so two records
that differ only in the order of their edges are equal and save alike. An
edge listed twice stays twice, and loading reports it (``repeated_edge``).
Several edges into one input are alternatives, opposite branches of one
decision, so nothing depends on the order of the list.

``Graph.compile`` compiles a record into a ``Graph``; what the record says that
can't be meant (a repeated edge, an edge naming a node the graph doesn't
have, an input set two ways) is a problem of that graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from pydantic import AfterValidator, ConfigDict, Field, JsonValue, ValidationError, field_validator
from pydantic_core import PydanticSerializationError, from_json, to_json

from conductor._internal.address import split_address
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.problem import InvalidCallError, Problem
from conductor._internal.problem import catalogue as problem
from conductor._internal.series import SET_HAS_NO_ORDER
from conductor._internal.value_type import kind_of_value


def _checked_address(address: str) -> str:
    """``address`` once it reads ``node.field``, or ``node.output.field`` for one field of a record an output
    holds; that field's own name may hold dots."""
    node_id, rest = split_address(address)
    field, dot, of_record = rest.partition(".")
    if not node_id or not field or (dot and not of_record):
        raise ValueError(f"{address!r} is not an address; write 'node.field', or 'node.output.field' for one field of a record")
    return address


#: An address, ``"node.field"`` or ``"node.output.field"`` for one field of a record an output holds: node ids and
#: a node's field names hold no dot, so everything after the second is the record's field (``"split.result.Amount in USD."``).
Address = Annotated[str, AfterValidator(_checked_address)]


def _empty(value: Mapping[str, Any]) -> bool:
    return not value


#: A record holds JSON, which has no float that isn't finite: one is refused, so a record equals what its JSON reads back.
_JSON_ONLY = ConfigDict(allow_inf_nan=False)


class NodeRecord(ConductorModel):
    """One node of a graph record: which definition it uses, and what is typed into its inputs.

    ``id`` names this use of the definition within the graph, and edges and
    addresses say it; ``node`` is the definition's id. The node's inputs and
    outputs are what loading answers (``graph.describe()``), not stored here.
    """

    model_config = ConductorModel.model_config | _JSON_ONLY

    id: str
    #: The id of the node definition this node uses.
    node: str
    #: Left out of the saved record when it is 1, as is each field below when empty.
    version: int = Field(default=1, exclude_if=lambda version: version == 1)
    #: The constants of these inputs: each has this value on every call in every run, and no caller may change it.
    constants: Mapping[str, JsonValue] = Field(default_factory=dict, exclude_if=_empty)
    #: Typed into these inputs, which stay open for a caller to change.
    defaults: Mapping[str, JsonValue] = Field(default_factory=dict, exclude_if=_empty)
    #: The host's own: position, size, titles a person edited. Stored and never read here.
    display: Mapping[str, JsonValue] = Field(default_factory=dict, exclude_if=_empty)


class GraphRecord(ConductorModel):
    """A graph as a host stores it: its nodes and its edges.

    What a graph takes and returns is derived when it is loaded, never
    stored: the open inputs of the nodes nothing feeds, and the outputs of the
    nodes nothing reads. What a host keeps about the whole graph (its name, a
    viewport) it stores beside the record.
    """

    model_config = ConductorModel.model_config | _JSON_ONLY

    #: Any sequence going in; a tuple once built.
    nodes: Sequence[NodeRecord] = ()
    #: ``(from, to)`` pairs of addresses, ``("read.text", "up.text")``; sorted once built.
    edges: Sequence[tuple[Address, Address]] = ()

    @field_validator("nodes")
    @classmethod
    def _nodes_as_tuple(cls, nodes: Sequence[NodeRecord]) -> tuple[NodeRecord, ...]:
        return tuple(nodes)

    @field_validator("edges")
    @classmethod
    def _edges_sorted(cls, edges: Sequence[tuple[str, str]]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(edges))


def as_record(record: GraphRecord | str | bytes) -> GraphRecord:
    """Return ``record`` when it is a ``GraphRecord``, else read it as a record's JSON text.

    A host that holds a dict makes the record itself: ``GraphRecord.model_validate(data)``.

    Raises:
        InvalidCallError: The text isn't JSON, or isn't a record's shape (a key a record doesn't have, a field
            of the wrong type, an address that isn't one): one ``invalid_record`` problem per fault, each at
            ``("record", *where)``. A record of the right shape that makes no sense as a graph is not raised
            here; loading it lists its problems.
    """
    if isinstance(record, GraphRecord):
        return record
    try:
        return GraphRecord.model_validate_json(record)
    except ValidationError as invalid:
        raise InvalidCallError(not_a_record(invalid)) from None


def not_a_record(invalid: ValidationError) -> list[Problem]:
    """Gives an ``invalid_record`` problem for each fault pydantic found reading a record, at ``("record", *where)``."""
    return [
        problem.invalid_record("record", *error["loc"], where=".".join(map(str, error["loc"])) or "the record", reason=error["msg"].removeprefix("Value error, "), error=error["type"])
        for error in invalid.errors(include_url=False, include_input=False)  # a validator's own sentence, without pydantic's prefix
    ]


def values_as_json(node_id: str, values: Mapping[str, Any]) -> tuple[dict[str, JsonValue], list[Problem]]:
    """``values`` typed into the node ``node_id`` as its record keeps them, each as its JSON reads back (a date as its ISO
    text, a float that isn't finite as ``"NaN"`` or ``"Infinity"``); and an ``invalid_value`` for each with no JSON form,
    or holding a set, whose JSON would list it in another order in another process (``set_has_no_order``)."""
    typed: dict[str, JsonValue] = {}
    problems: list[Problem] = []
    for name, value in values.items():
        if _holds_a_set(value):
            errors: list[dict[str, JsonValue]] = [{"type": "set_has_no_order", "loc": [], "msg": SET_HAS_NO_ORDER}]
            problems.append(problem.invalid_value(node_id, name, reason=SET_HAS_NO_ORDER, errors=errors))
            continue
        try:
            typed[name] = from_json(to_json(value, inf_nan_mode="strings"))
        except PydanticSerializationError:
            reason = "a value typed into a graph is saved with it as JSON" + kind_of_value(value).typed_in_advice
            problems.append(problem.invalid_value(node_id, name, reason=reason))
    return typed, problems


def _holds_a_set(value: Any) -> bool:
    if isinstance(value, (set, frozenset)):
        return True
    if isinstance(value, dict):
        return any(map(_holds_a_set, value.values()))
    return isinstance(value, (list, tuple)) and any(map(_holds_a_set, value))
