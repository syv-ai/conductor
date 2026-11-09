"""``Graph`` and ``GraphNode`` — the persisted graph.

A graph is a list of nodes and nothing else: edges live in each
node's ``bindings``, and a canvas derives its own edges from them. A
node's fields fall in three groups:

* **behaviour** — ``type``, ``version``, ``bindings``, ``locked`` — is what
  the runtime reads and branches on;
* **content** — the node's ``title``, ``description`` and one
  ``FieldContent`` per field — is shown to people and emitted in events,
  never branched on;
* **display** — ``display`` — is stored and returned and never parsed.

So a diff of the behaviour fields is the behavioural diff, and display can
change shape without a migration. Each record is a ``ConductorModel``,
so a graph saves and loads itself (``Graph.from_path``, ``graph.to_yaml()``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import Field, field_validator

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.graph.binding import Binding


class FieldContent(ConductorModel):
    """The title and description a person reads for one field of one node.

    Copied from the node's declaration when the node is added, and edited
    freely afterwards; nothing resolves it against the declaration again,
    so an upstream rename never reaches an existing graph. Read by
    whatever shows the field to a person, by nothing that decides
    behaviour.
    """

    title: str
    description: str | None = None


class GraphNode(ConductorModel):
    """One node as it sits in a graph: one use of a definition.

    A definition (``NodeDefinition``) is what a developer wrote; a node in
    a graph is one use of it, with its own ``id``, minted when the node is
    added and never renamed.

    ``type`` names the definition and ``version`` selects one of its
    versions; ``bindings`` says where each input's value comes from (one
    entry per input, so a static and an edge cannot both claim one);
    ``locked`` names inputs no caller may fill. ``title``, ``description``
    and ``fields`` are copies of the declaration's text that the author may
    edit. ``display`` is the canvas's own::

        GraphNode(
            id="upper-1", type="upper", version=1,
            bindings={"text": From("reader-1.text")},
            title="Upper case",
        )

    A canvas derives edges from ``bindings``. What the node's inputs and outputs actually are is
    answered when the graph is compiled (``CompiledGraph.node(node_id).interface``), not stored here.
    """

    id: str
    #: Which definition runs this node, and which of its versions. Two
    #: fields; nothing parses a ``type@version`` string.
    type: str
    version: int
    bindings: Mapping[str, Binding] = Field(default_factory=dict)
    #: Inputs of this node no caller may fill. Only matters on a
    #: node with no edge into it, since only those offer inputs to a
    #: caller.
    locked: tuple[str, ...] = ()
    #: Content.
    title: str = ""
    description: str = ""
    fields: Mapping[str, FieldContent] = Field(default_factory=dict)
    #: Display: position, size, whatever the canvas keeps. Never parsed here.
    display: Mapping[str, Any] = Field(default_factory=dict)


class Graph(ConductorModel):
    """The persisted graph: its nodes and its display.

    What a host stores for one version of a graph; edges live in each
    node's ``bindings``. What the graph takes and returns is derived when it is
    compiled (``derive_interface``), never stored.
    """

    #: Any sequence going in; a tuple once built.
    nodes: Sequence[GraphNode]
    #: Display, for the whole graph.
    display: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("nodes")
    @classmethod
    def _nodes_as_tuple(cls, nodes: Sequence[GraphNode]) -> tuple[GraphNode, ...]:
        return tuple(nodes)
