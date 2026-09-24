"""``RunState`` — a frozen snapshot of what a run has produced so far, as a host stores it between legs.

A run takes several legs when a node waits on a person: one call of
``execute`` is a leg, and the next leg has to start from everything the
earlier ones produced. The run's state is that everything, in a form a
host can keep in a row and hand back: every value the ledger holds, in
JSON form, which units are done, and a fingerprint per node of how the
graph placed it. Nothing else: which rows each index has, which indexes
will gain no more and where one has none all follow from the values and
the done units, and a restore works them out again. It accumulates across legs: each leg starts from
the state the last one ended on and ends on a new one.

Written by ``Ledger.state()`` at the end of every leg and carried on the
ending event as ``state``; read by ``Ledger.restore`` when a host passes
it to ``execute(state=...)``. The fingerprints are what make a state safe
to restore into a graph the author has since edited: a node whose
fingerprint differs from the graph's — a static edited, a version bumped,
a binding moved — is dropped with everything downstream of it and runs
again, and so is a node the graph no longer has.

Not the results: ``state.results(compiled)`` is what the run produced by
node and output, as values; the state is what the ledger needs to go on,
in wire form, and a host never reads inside it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from conductor.model import ConductorModel
from conductor.ref import Ref
from conductor.series import Row

if TYPE_CHECKING:
    from conductor.graph.compiled import CompiledGraph


class StateEntry(ConductorModel):
    """Where one entry of a run's state sits: an output of a node, by its
    address (``ref``, ``"node.field"``), at a row (``None`` on a node that
    ran once). What sits there is the subclass's: ``StateValue`` or
    ``StateSkip``."""

    ref: Ref
    row: Row | None


class StateValue(StateEntry):
    """A value the run holds there, as the codec wrote it (which may be
    ``null``). Its twin is ``StateSkip``: an entry is one or the other, told
    apart by which key it has."""

    value: Any


class StateSkip(StateEntry):
    """A skip the run holds there in place of a value: ``skipped`` is the
    depth of the row the skip was written at. A skip has no type, so nothing
    is written for a value."""

    skipped: int


class DoneUnit(ConductorModel):
    """A unit that ran to completion: a node, and the row it ran for
    (``None`` for a node that ran once)."""

    node_id: str
    row: Row | None


class RunState(ConductorModel):
    """A frozen snapshot of a run's state: the ledger in wire form, plus a fingerprint per node.

    A snapshot, not live state: the ledger a leg runs on is the live thing,
    and this is what it wrote down when the leg ended. Nothing changes it
    in place; ``without`` returns a new one.

    ``values`` are what each output holds at each row: a ``StateValue``, or a
    ``StateSkip`` where the output was skipped. An entry that is neither is
    refused when the state is read, so a malformed state fails where it
    arrives rather than inside a restore.
    ``done_units`` are the units that ran to completion, each a ``DoneUnit``. ``node_fingerprints`` is
    one hash per node of its placement in the graph,
    ``CompiledNode.fingerprint``. ``RunState()`` is the state of a run that
    has produced nothing.
    """

    values: list[StateValue | StateSkip] = Field(default_factory=list)
    done_units: list[DoneUnit] = Field(default_factory=list)
    node_fingerprints: dict[str, str] = Field(default_factory=dict)

    def results(self, compiled: CompiledGraph) -> dict[str, dict[str, Any]]:
        """What the run produced: every complete node's outputs, typed, by expanded node id.

        Read over ``compiled``, the graph the run is of, the way the next leg
        would read this state: a node the graph has changed since, and
        everything reading it, is left out. A node that did not run is
        absent; one that ran per row is a ``Series``, sparse where rows were
        skipped. The state an ending just carried is read off the ledger
        that wrote it, with nothing decoded; a stored one is decoded
        through the codec by each field's type.
        """
        from conductor.execution.ledger import Ledger

        return Ledger.results_of(compiled, self)

    def without(self, *node_ids: str) -> RunState:
        """This state as if these nodes had never run: a host's "run from here".

        Their values and done units go, and so do their fingerprints,
        which is what tells ``Ledger.restore`` to run them again with
        everything that reads them. A node id also covers the inner nodes of
        an embedded graph placed under it (``approve`` covers ``approve/check``).
        """
        def gone(node_id: str) -> bool:
            return any(node_id == dropped or node_id.startswith(f"{dropped}/") for dropped in node_ids)

        return self.model_copy(update={
            "values": [entry for entry in self.values if not gone(entry.ref.node_id)],
            "done_units": [unit for unit in self.done_units if not gone(unit.node_id)],
            "node_fingerprints": {node_id: fingerprint for node_id, fingerprint in self.node_fingerprints.items() if not gone(node_id)},
        })
