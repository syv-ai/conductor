"""``decision`` — route a value to one of two branches.

The node sends ``value`` to its ``if_true`` or ``if_false`` output
according to ``when`` and returns ``SKIPPED`` on the other, so whatever is
wired to the branch not taken does not run. The engine treats ``SKIPPED``
like any other value; the node has no special role. A choice between more
than two branches is a chain of decisions.

``when`` is a ``Flag``, not an expression to parse: the catalog already
produces booleans (``text-contains``, ``regex-match``, ``logic-not``), so
the condition is wired in rather than written in a language this node
would have to own.

``value`` is declared ``Any`` because the node routes it without reading
it and so cannot name its type. ``compute_outputs`` copies the type that
arrives on ``value`` onto both outputs. The two outputs share the
``choice`` ``"when"``: exactly one of them is produced per run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Annotated, Any

from conductor._sentinel import SKIPPED
from conductor.metadata import Output
from conductor.returns import Result
from conductor.widgets import ConnectionList, Switch

from conductor_nodes.types import Flag, StdlibNode

if TYPE_CHECKING:
    from conductor import NodeRegistry
    from conductor.dtype import DType


@dataclass(frozen=True)
class Branches:
    """What ``Decision.run`` returns: one field per output.

    Both fields share ``choice="when"``, so the interface records that
    exactly one of them is produced; the other holds ``SKIPPED``.
    """

    if_true: Annotated[Any, Result(title="If true", choice="when")]
    if_false: Annotated[Any, Result(title="If false", choice="when")]


class Decision(StdlibNode):
    id = "decision"
    title = "Decision"
    description = "Routes a value to one of two branches according to `when`."
    category = "control"

    def run(
        self,
        value: Annotated[Any, ConnectionList(title="Value")],
        when: Annotated[Flag, Switch(title="When")] = Flag(True),
    ) -> Branches:
        if when:
            return Branches(if_true=value, if_false=SKIPPED)
        return Branches(if_true=SKIPPED, if_false=value)

    def compute_outputs(
        self,
        declared: tuple[Output, ...],
        values: Mapping[str, Any],
        arriving: Mapping[str, "type[DType]"],
    ) -> tuple[Output, ...]:
        """Give both outputs the type that arrives on ``value`` (``Any`` until wired)."""
        dtype = arriving.get("value", Any)
        return tuple(replace(out, dtype=dtype) for out in declared)


NODES = (Decision,)


def register(registry: "NodeRegistry") -> None:
    """Register the ``decision`` node on ``registry``."""
    for node_cls in NODES:
        registry.register(node_cls)
