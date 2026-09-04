"""``Flow`` and ``GraphNode`` — the persisted graph.

A flow is a list of placed nodes and nothing else: wiring lives in each
node's ``bindings``, and a canvas derives its own cables from them. A
placement's fields fall in three groups:

* **behaviour** — ``type``, ``version``, ``bindings``, ``locked`` — is what
  the runtime reads and branches on;
* **content** — the placement's ``title``, ``description`` and one
  ``FieldContent`` per field — is shown to people and emitted in events,
  never branched on;
* **chrome** — ``display`` — is stored and returned and never parsed.

So a diff of the behaviour fields is the behavioural diff, and chrome can
change shape without a migration. A host stores these records through
pydantic.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from conductor.graph.binding import Binding, static_values


@dataclass(frozen=True)
class FieldContent:
    """The title and description a person reads for one field of one placement.

    Copied from the node's declaration when the node is placed, and edited
    freely afterwards; nothing resolves it against the declaration again,
    so an upstream rename never reaches an existing graph. Read by
    whatever shows the field to a person, by nothing that decides
    behaviour.
    """

    title: str
    description: str | None = None


@dataclass(frozen=True)
class GraphNode:
    """One placement of a node in a flow.

    ``type`` names the definition and ``version`` selects one of its
    versions; ``bindings`` says where each input's value comes from (one
    entry per input, so a typed value and a cable cannot both claim one);
    ``locked`` names inputs no caller may fill. ``title``, ``description``
    and ``fields`` are copies of the declaration's text that the author may
    edit. ``display`` is the canvas's own::

        GraphNode(
            id="upper-1", type="upper", version=1,
            bindings={"text": Sources(refs=(Ref("reader-1", "text"),))},
            title="Upper case",
        )

    There is no edge record beside it; a canvas derives cables from
    ``bindings``. What the placement's inputs and outputs actually are
    is answered when the flow is compiled, not stored here.
    """

    id: str
    #: Which definition runs this node, and which of its versions. Two
    #: fields; nothing parses a ``type@version`` string.
    type: str
    version: int
    bindings: Mapping[str, Binding] = field(default_factory=dict)
    #: Inputs of this placement no caller may fill. Only matters on a
    #: placement with no wire into it, since only those offer inputs to a
    #: caller.
    locked: tuple[str, ...] = ()
    #: Content.
    title: str = ""
    description: str = ""
    fields: Mapping[str, FieldContent] = field(default_factory=dict)
    #: Chrome: position, size, whatever the canvas keeps. Never parsed here.
    display: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # ``.`` separates node from field in a ``Ref``, so an id may not
        # contain one. ``/`` may: the compiler uses it to namespace an
        # embedded flow's nodes under the placing node's id (``approve/check``).
        if "." in self.id:
            raise ValueError(f"node id {self.id!r} contains '.': a Ref reads as 'node.field' and must read one way")

    @property
    def data(self) -> dict[str, Any]:
        """The typed-in values as a plain dict, for the engine."""
        return static_values(self.bindings)


@dataclass(frozen=True)
class Flow:
    """A flow: its placed nodes and its chrome. The whole persisted graph.

    There is no edge list, no trigger and no settings block. What the
    flow takes and returns is derived when it is compiled
    (``derive_interface``), never stored.
    """

    nodes: list[GraphNode]
    #: Chrome, at flow level.
    display: Mapping[str, Any] = field(default_factory=dict)

