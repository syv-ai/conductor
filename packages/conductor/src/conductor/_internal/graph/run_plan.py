"""``compiled.run_plan``: where each node runs and what each of its inputs receives, for a person to read.

An author draws edges and never says "for each". Compile decides, and this is
what it decided, in terms a host can show and compare::

    print(compiled.run_plan)
    # upload  once
    # split   per row of upload.result  text: the value at my row (upload.result)
    # count   per row of upload.result  lines: all rows of split.result under my row; none under it: called with an empty list

Built on the first read, from the same records the engine runs on, so what it
says is what happens.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.graph.receive_modes import (
    Broadcast,
    EntireValue,
    Gather,
    Merge,
    PerRow,
    RowsUnder,
)
from conductor._internal.ref import Ref

if TYPE_CHECKING:
    from conductor._internal.graph.compiled_node import CompiledNode
    from conductor._internal.series import Index

#: How a value reaches one call of a node.
#:
#: ``value_at_row``: the value at the call's own row. ``same_value_every_row``:
#: one value, copied to every call; also a static or a default.
#: ``all_rows_under_row``: every row of a column under the call's row, however
#: deep, flat. ``entire_value``: everything on the field, once. ``collected``: unrelated
#: sources gathered into one series. ``merged``: tables and columns matched on
#: their rows' ids into one table (``source_plans`` says how each arrives).
ReceiveModeName = Literal["value_at_row", "same_value_every_row", "all_rows_under_row", "entire_value", "collected", "merged"]


class InputPlan(ConductorModel):
    """How one input of one node is filled."""

    #: The outputs wired to it; empty for a static or a default.
    sources: tuple[Ref, ...]
    receive_mode: ReceiveModeName
    #: The field whose rows it reads: for ``value_at_row`` and ``all_rows_under_row``. ``None`` otherwise.
    rows_of: Ref | None = None
    #: For ``all_rows_under_row``: the index the rows are grouped under, which a
    #: call's row sits on. Equal to ``rows_of`` when a call gets its own row's cell.
    grouped_under: Ref | None = None
    #: A blank here calls the node with ``None`` (or keeps the cell, on a series). If not, the row isn't called.
    takes_blank: bool
    #: For ``merged``: one plan per source, in wiring order.
    source_plans: tuple[InputPlan, ...] = ()
    #: On a part of a merge: a parent table or column whose value is copied down
    #: onto the rows of a deeper source.
    copied_down: bool = False

    def __str__(self) -> str:
        source = " + ".join(self.sources)
        match self.receive_mode:
            case "value_at_row":
                return f"the value at my row ({self.rows_of})"
            case "same_value_every_row":
                return f"the same value every row ({source})" if source else "the same value every row"
            case "all_rows_under_row":
                grouped_under = "" if self.grouped_under in (None, self.rows_of) else f" (grouped under {self.grouped_under})"
                return f"all rows of {self.rows_of} under my row{grouped_under}; none under it: called with an empty list"
            case "entire_value":
                return f"all of {source}" if source else "the whole list set on the node"
            case "collected":
                return f"collected into one list: {source}"
            case "merged":
                return "merged: " + " + ".join(f"{part.sources[0]}{' [copied down]' if part.copied_down else ''}" for part in self.source_plans)


class NodeRunPlan(ConductorModel):
    """Where one node runs and what each of its inputs receives."""

    node_id: str
    #: The field whose rows this node runs once per row of; ``None`` when it runs once.
    runs_per_row_of: Ref | None
    inputs: dict[str, InputPlan]
    #: The outputs of this node that make rows.
    makes_rows: tuple[Ref, ...]

    def _as_text_line(self, width: int) -> str:
        where = "once" if self.runs_per_row_of is None else f"per row of {self.runs_per_row_of}"
        shown = [f"{name}: {plan}" for name, plan in self.inputs.items() if plan.sources or plan.rows_of is not None]
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


def _plan_of_receive_mode(sources: tuple[Ref, ...], receive_mode: object, takes_blank: bool) -> InputPlan:
    """What a receive mode means, in the terms a host sees."""
    match receive_mode:
        case PerRow(index=index):
            return InputPlan.model_construct(sources=sources, receive_mode="value_at_row", rows_of=_index_as_address(index), grouped_under=None, takes_blank=takes_blank, source_plans=(), copied_down=False)
        case Broadcast():
            return InputPlan.model_construct(sources=sources, receive_mode="same_value_every_row", rows_of=None, grouped_under=None, takes_blank=takes_blank, source_plans=(), copied_down=False)
        case EntireValue():
            return InputPlan.model_construct(sources=sources, receive_mode="entire_value", rows_of=None, grouped_under=None, takes_blank=takes_blank, source_plans=(), copied_down=False)
        case RowsUnder(index=index, depth=depth):
            return InputPlan.model_construct(sources=sources, receive_mode="all_rows_under_row", rows_of=_index_as_address(index), grouped_under=Ref(index.lineage[depth - 1].id), takes_blank=takes_blank, source_plans=(), copied_down=False)
        case Gather():
            return InputPlan.model_construct(sources=sources, receive_mode="collected", rows_of=None, grouped_under=None, takes_blank=takes_blank, source_plans=(), copied_down=False)
    raise AssertionError(f"no plan for {receive_mode!r}")


def plan_of_node(node: CompiledNode) -> NodeRunPlan:
    """The run plan of one ready node."""
    assert node.wiring is not None and node.interface is not None
    fields = node.wiring.fields
    inputs: dict[str, InputPlan] = {}
    for input_ in node.interface.inputs:
        compiled_input = fields[input_.name]
        binding = compiled_input.binding
        sources = tuple(binding.refs) if binding is not None and hasattr(binding, "refs") else ()
        receive_mode = compiled_input._receive_mode
        if isinstance(receive_mode, Merge):
            source_plans = tuple(
                _plan_of_receive_mode((source,), part, input_.takes_blank).model_copy(update={"copied_down": source in receive_mode.copied_down})
                for source, part in receive_mode.receive_mode_by_source
            )
            inputs[input_.name] = InputPlan.model_construct(sources=sources, receive_mode="merged", rows_of=None, grouped_under=None, takes_blank=input_.takes_blank, source_plans=source_plans, copied_down=False)
        else:
            inputs[input_.name] = _plan_of_receive_mode(sources, receive_mode, input_.takes_blank)
    runs_per_row_of = node.wiring._runs_per_row_of
    return NodeRunPlan.model_construct(
        node_id=node.id,
        runs_per_row_of=_index_as_address(runs_per_row_of),
        inputs=inputs,
        makes_rows=tuple(field.ref for field in fields.values() if field._makes_rows is not None),
    )
