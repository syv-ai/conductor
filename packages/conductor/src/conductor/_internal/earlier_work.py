"""The work of an earlier run that a run may take in place of calling a node."""

from __future__ import annotations

from dataclasses import dataclass, field

from conductor._internal.address import Key
from conductor._internal.units import Answered, Asking, Done


@dataclass(slots=True)
class Earlier:
    """The work of one node that a run may take from the run it went on from, in place of calling the node.

    A person's answer, or a question still waiting, is taken once, by the next unit in order that received the
    same under the same items above it. A node's own result is taken by the unit at the same place when it
    received the same there, and else by what was received, whichever unit made it.

    Attributes:
        at: Each result by the instance its unit stood in, with the fingerprint of what it received.
        computed: A node's results by the fingerprint of what the unit received.
        asked: A person's answers and the questions still waiting, by the fingerprint of what the unit received and
            of the items above it, in the order they stood.
    """

    at: dict[Key, tuple[str, Done]] = field(default_factory=dict[Key, tuple[str, Done]])
    computed: dict[str, Done] = field(default_factory=dict[str, Done])
    asked: dict[tuple[str, str], list[Answered | Asking]] = field(default_factory=dict[tuple[str, str], list["Answered | Asking"]])
