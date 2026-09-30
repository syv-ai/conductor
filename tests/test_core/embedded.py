"""A test helper: a stored graph offered as a node, the way a host offers one."""

from typing import Any, ClassVar

from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.node import NodeDefinition
from conductor.registry import NodeRegistry


def embedded_graph_node(node_id: str, graph: tuple[GraphNode, ...], registry: NodeRegistry, graph_title: str = "Embedded") -> type[NodeDefinition]:
    """A definition whose one version is ``graph``, compiled once against ``registry``."""

    class Embedded(NodeDefinition):
        id = node_id
        title = graph_title
        description = "d"
        category = "test"
        versions: ClassVar[dict[int, Any]] = {1: CompiledGraph.from_graph(Graph(nodes=list(graph)), registry)}

    return Embedded
