"""``NodeRecord``: one node of a graph record, ``{"id": "shout", "node": "shout", "constants": {"mark": "?"}}``."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ConfigDict, Field, JsonValue

from conductor._internal.base import ConductorModel


def _empty(value: Mapping[str, Any]) -> bool:
    return not value


#: A record holds JSON, which has no float that isn't finite: one is refused, so a record equals what its JSON reads back.
JSON_ONLY = ConfigDict(allow_inf_nan=False)


class NodeRecord(ConductorModel):
    """One node of a graph record: which node definition it uses, and the constants and defaults of its inputs.

    ``id`` names this use of the definition within the graph, and edges and addresses name the node by it; ``node``
    is the definition's id. The node's inputs and outputs are not stored here: compiling works them out
    (``graph.describe()``).
    """

    model_config = ConductorModel.model_config | JSON_ONLY

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
