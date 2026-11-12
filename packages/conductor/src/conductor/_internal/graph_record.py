"""``GraphRecord`` and ``NodeRecord``: a graph as it is stored.

A graph record is its nodes and its edges, and nothing else::

    {
        "nodes": [
            {"id": "lines", "node": "lines"},
            {"id": "shout", "node": "shout", "constants": {"mark": "?"}},
            {"id": "join", "node": "join"}
        ],
        "edges": [["lines.result", "shout.line"], ["shout.result", "join.texts"]]
    }

A node names the node definition it uses (``node``) and its version (``version``, 1 when left out). A constant
(``constants``) is the value an input has on every call in every run: no caller can change it. A default
(``defaults``) is the value of a graph input when the caller gives none. ``display`` is the host's own: conductor
stores it and never reads it. An empty field is left out when the record is saved, so a bare node is
``{"id": "join", "node": "join"}``.

Edges are a list of ``[source, target]`` addresses, ``node.input`` or ``node.output``, or ``node.output.field`` for
one field of a record an output gives. They are kept sorted, so two records that differ only in the order of their
edges are equal and save alike. An edge listed twice stays twice, and compiling reports it (``repeated_edge``).
Several edges into one input are alternatives, from opposite branches of one decision, so nothing depends on the
order of the list.

A host hands a record in as a ``GraphRecord`` or its JSON text; ``gate/record.py`` reads it. ``Graph.compile``
compiles it into a ``Graph``, and what the record says that cannot be meant (a repeated edge, an edge naming a node
the graph does not have, an input given both a constant and a default) is a problem of that graph.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from pydantic import AfterValidator, ConfigDict, Field, JsonValue, field_validator

from conductor._internal.address import split_address
from conductor._internal.conductor_model import ConductorModel


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
    """One node of a graph record: which node definition it uses, and the constants and defaults of its inputs.

    ``id`` names this use of the definition within the graph, and edges and addresses name the node by it; ``node``
    is the definition's id. The node's inputs and outputs are not stored here: compiling works them out
    (``graph.describe()``).
    """

    model_config = ConductorModel.model_config | _JSON_ONLY

    id: str
    #: The id of the node definition this node uses.
    node: str
    #: Left out of the saved record when it is 1, as is each field below when empty.
    version: int = Field(default=1, exclude_if=lambda version: version == 1)
    #: The constants of these inputs: each has this value on every call in every run, and no caller may change it.
    constants: Mapping[str, JsonValue] = Field(default_factory=dict, exclude_if=_empty)
    #: The defaults of these inputs: a graph input takes its default when the caller gives no value.
    defaults: Mapping[str, JsonValue] = Field(default_factory=dict, exclude_if=_empty)
    #: The host's own: position, size, titles a person edited. Stored and never read here.
    display: Mapping[str, JsonValue] = Field(default_factory=dict, exclude_if=_empty)


class GraphRecord(ConductorModel):
    """A graph as a host stores it: its nodes and its edges.

    What a graph takes and returns is worked out when it is compiled, never stored: its graph inputs, the inputs
    with no edge on the nodes no edge leads into, and its graph outputs, the outputs of the nodes no edge leaves.
    What a host keeps about the whole graph (its name, a viewport) it stores beside the record.
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
