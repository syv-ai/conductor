"""A node as an editor's palette shows it: ``NodeDescription``, one record for a function and a subgraph alike.
``shout.describe()`` is ``NodeDescription(id="shout", ...)``.
"""

from __future__ import annotations

from pydantic import JsonValue

from conductor._internal.base import ConductorModel
from conductor._internal.field import FieldDescription

from .policy import Deprecation


class NodeDescription(ConductorModel):
    """A node as an editor's palette shows it, whether its author wrote a function or a graph record.

    A palette offers a node to pick and wire, so this is what a person and a host need for that: what the node is,
    its fields, and whether a run of it can wait for a person (``asks``). How the engine calls a function, its
    ``Policy``, stays on the definition: a subgraph is never called, and each node inside it runs by its own policy.

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
        asks: Whether a call may wait for a person: a function that raises ``Ask``, or a subgraph with such a node
            inside it, at any depth. A graph with such a node can pause, so a host can't run it unattended.
        takes_added_inputs: Whether what is typed into it adds inputs; never for a subgraph.
        computes_outputs: Whether its outputs depend on what is typed into it and on what arrives; never for a
            subgraph.
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
    asks: bool
    takes_added_inputs: bool
    computes_outputs: bool
