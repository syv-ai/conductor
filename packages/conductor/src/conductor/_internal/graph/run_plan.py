"""``compiled.run_plan``: where each node runs and what each of its inputs receives, for a person to read.

An author draws edges and never says "for each". Compile decides, and this is
what it decided, in terms a host can show and compare::

    print(compiled.run_plan)
    # upload  once
    # split   per row of upload.result  text: the value at my row (upload.result)
    # count   per row of upload.result  lines: all rows of split.result under my row; none under it: called with an empty list

Each input's entry is its ``ReceiveMode`` (``conductor._internal.graph.receive_modes``),
the record the run reads, so what it says is what happens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.graph.receive_modes import ReceiveMode
from conductor._internal.ref import Ref

if TYPE_CHECKING:
    from conductor._internal.graph.compiled_node import CompiledNode
    from conductor._internal.series import Index


class NodeRunPlan(ConductorModel):
    """Where one node runs and what each of its inputs receives."""

    node_id: str
    #: The field whose rows this node runs once per row of; ``None`` when it runs once.
    runs_per_row_of: Ref | None
    inputs: dict[str, ReceiveMode]
    #: The outputs of this node that make rows.
    makes_rows: tuple[Ref, ...]

    def _as_text_line(self, width: int) -> str:
        where = "once" if self.runs_per_row_of is None else f"per row of {self.runs_per_row_of}"
        shown = [f"{name}: {plan}" for name, plan in self.inputs.items() if plan.sources]
        return f"{self.node_id:<{width}}  {where:<28}  {shown[0] if shown else ''}".rstrip() + "".join(
            f"\n{'':<{width}}  {'':<28}  {more}" for more in shown[1:]
        )


class RunPlan(ConductorModel):
    """Where every ready node runs and what each of its inputs receives, in
    execution order. ``str(plan)`` is that as text, one node to a line."""

    nodes: dict[str, NodeRunPlan]

    def __str__(self) -> str:
        width = max((len(node_id) for node_id in self.nodes), default=0)
        return "\n".join(plan._as_text_line(width) for plan in self.nodes.values())


def _index_as_address(index: Index | None) -> Ref | None:
    return None if index is None else Ref(index.id)


def plan_of_node(node: CompiledNode) -> NodeRunPlan:
    """The run plan of one ready node."""
    assert node.wiring is not None and node.interface is not None
    fields = node.wiring.fields
    return NodeRunPlan.model_construct(
        node_id=node.id,
        runs_per_row_of=_index_as_address(node.wiring._runs_per_row_of),
        inputs={input_.name: fields[input_.name]._receive_mode for input_ in node.interface.inputs},
        makes_rows=tuple(field.ref for field in fields.values() if field._makes_rows is not None),
    )
