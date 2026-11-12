"""A node as an editor's palette shows it: ``FunctionNodeDescription`` and ``SubgraphDescription``, told apart by
``kind``, the two kinds of ``NodeDescription``. ``shout.describe()`` is ``FunctionNodeDescription(id="shout", ...)``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field as PydanticField
from pydantic import JsonValue

from conductor._internal.base import ConductorModel
from conductor._internal.field import FieldDescription
from conductor._internal.node.policy import Deprecation, Policy


class _NodeDescription(ConductorModel):
    """What an editor's palette shows of every node. Never made on its own: a node is described as a function or as a
    subgraph.

    Attributes:
        id: The id a graph names the node by.
        version: The version described, the newest.
        versions: Every version there is, newest first.
        title: What a person reads as its name.
        description: What it does, for a person who is no developer.
        category: The group a palette lists it under.
        tags: Words a palette filters by.
        extra: What the host stored on the node, which conductor never reads.
        docs: A longer text or a link about the node.
        deprecation: The notice that this version is going away, when it is.
        inputs: Its inputs.
        outputs: Its outputs.
    """

    id: str
    version: int
    versions: list[int]
    title: str
    description: str
    category: str
    tags: list[str]
    extra: dict[str, JsonValue]
    docs: str | None
    deprecation: Deprecation | None
    inputs: list[FieldDescription]
    outputs: list[FieldDescription]


class FunctionNodeDescription(_NodeDescription):
    """A node that is a function, as an editor's palette shows it.

    Attributes:
        kind: ``"function"``.
        policy: How the engine calls it.
        takes_added_inputs: Whether what is typed into it adds inputs.
        computes_outputs: Whether its outputs depend on what is typed into it and on what arrives.
        asks: Whether a call may wait for a person: its function raises ``Ask``, or it has an ``Asked`` parameter. A
            graph with such a node can pause, so a host can't run it unattended.
        asked: The ``Asked`` parameter a person may be asked to fill, drawn like an input though no edge reaches it;
            ``None`` when the node has none.
    """

    kind: Literal["function"] = "function"
    policy: Policy
    takes_added_inputs: bool
    computes_outputs: bool
    asks: bool
    asked: FieldDescription | None


class SubgraphDescription(_NodeDescription):
    """A subgraph as an editor's palette shows it.

    Its inputs and outputs are the graph inputs and graph outputs of its graph record, so only a registry that
    has the node definitions that record uses can describe it.

    Attributes:
        kind: ``"subgraph"``.
    """

    kind: Literal["subgraph"] = "subgraph"


#: A node as an editor's palette shows it, one of two kinds told apart by ``kind``.
NodeDescription = Annotated[FunctionNodeDescription | SubgraphDescription, PydanticField(discriminator="kind")]
