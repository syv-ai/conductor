"""``NodeDescription``, a node as an editor's palette shows it.

``shout.describe()`` gives ``NodeDescription(id="shout", ...)``. A function node and a subgraph are described the
same way.
"""

from __future__ import annotations

from pydantic import JsonValue

from conductor._internal.base import ConductorModel
from conductor._internal.field import FieldDescription

from .policy import Deprecation


class NodeDescription(ConductorModel):
    """A node as an editor's palette shows it, whether it is a function or a subgraph.

    A palette lets a person pick a node and wire it into a graph. So this has what that needs: what the node is,
    its inputs and outputs, and whether a call can wait for a person (``asks``). The node's ``Policy`` isn't here.
    It stays on the definition, since a subgraph is never called and has none.

    Attributes:
        id: The id a graph names the node by.
        version: The version described, which is the newest.
        versions: Every version there is, newest first.
        title: The node's name, as a person reads it.
        description: What the node does, for a person who isn't a developer.
        category: The group a palette lists the node under.
        tags: Words a palette can filter by.
        extra: The host's own data on the node. Conductor never looks at it.
        docs: A longer text or a link about the node.
        deprecation: The notice that this version is going away, if it is.
        inputs: The node's inputs.
        outputs: The node's outputs.
        asks: Whether a call may wait for a person. That is a function that raises ``Ask``, or a subgraph with such
            a node somewhere inside. A graph with such a node can pause, so a host can't run it unattended.
        takes_added_inputs: Whether the node's constants can add inputs. Never for a subgraph.
        computes_outputs: Whether the node's outputs depend on its constants and on the types its edges bring.
            Never for a subgraph.
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
