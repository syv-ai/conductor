"""``compiled.describe()``: a compiled graph as an editor draws it.

The twin of ``registry.describe()``. That one says what could be placed on a
canvas; this one says what compile made of a canvas: whether it runs, what is
wrong with it, what it takes and returns, where each node runs, and for each node
the author drew how far compile got and which fields it has.

Every key of ``fields`` is the address an author writes in an edge
(``"summary.text"``), a node that is a graph included (``"approve.amount"``),
and ``compiled.field`` reads each one. The record is written as JSON for a
reader in any language and read back by none: an ``Input`` holds a type, which
only a registry can name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.dtype import DTypeField
from conductor._internal.errors import Problem
from conductor._internal.graph.compiled_node import NodeStatus
from conductor._internal.graph.run_plan import RunPlan
from conductor._internal.node_signature import Interface
from conductor._internal.ref import Ref

if TYPE_CHECKING:
    from conductor._internal.graph.compiled_graph import CompiledGraph


class FieldDescription(ConductorModel):
    """One input or output of a ``ready`` node, as compile typed it."""

    #: The type that travels on it.
    dtype: DTypeField
    #: Whether it holds many values rather than one: its results are rows of values.
    many: bool


class CompiledNodeDescription(ConductorModel):
    """One node the author drew, as compile left it."""

    status: NodeStatus
    #: Its inputs and outputs, with the types its edges gave them; ``None`` on a ``resolution_failed`` node.
    interface: Interface | None


class GraphDescription(ConductorModel):
    """A compiled graph as an editor reads it. Built on demand by ``CompiledGraph.describe()``."""

    is_runnable: bool
    #: Everything wrong with the graph, each problem on the node and field it is about.
    problems: tuple[Problem, ...]
    #: What the graph takes and returns, named by address.
    interface: Interface
    #: Where each node runs and what its inputs receive; a node inside an embedded graph is at its path.
    run_plan: RunPlan
    #: Every node the author drew, by id.
    nodes: dict[str, CompiledNodeDescription]
    #: Every input and output of a ``ready`` node, by address.
    fields: dict[Ref, FieldDescription]


def describe_graph(compiled: CompiledGraph) -> GraphDescription:
    """The record of ``compiled``: one entry per node the author drew and per field of each."""
    nodes: dict[str, CompiledNodeDescription] = {}
    fields: dict[Ref, FieldDescription] = {}
    for drawn in compiled.graph.nodes:
        node = compiled.node(drawn.id)
        nodes[drawn.id] = CompiledNodeDescription(
            status=node.status,
            interface=node.interface,
        )
        if node.status == "ready":
            assert node.interface is not None
            names = [input_.name for input_ in node.interface.inputs] + [output.name for output in node.interface.outputs]
            for name in names:
                ref = Ref(drawn.id, name)
                field = compiled.field(ref)
                fields[ref] = FieldDescription(dtype=field.dtype, many=field.many)
    return GraphDescription(
        is_runnable=compiled.is_runnable,
        problems=compiled.problems,
        interface=compiled.interface,
        run_plan=compiled.run_plan,
        nodes=nodes,
        fields=fields,
    )
