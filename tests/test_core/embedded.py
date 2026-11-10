"""A test helper: a stored graph offered as a node, the way a host offers one."""


from conductor import CompiledGraph, NodeDefinition, NodeRegistry
from helpers import Placed, record


def embedded_graph_node(node_id: str, graph: tuple[Placed, ...], registry: NodeRegistry, graph_title: str = "Embedded") -> NodeDefinition:
    """A definition whose one version is ``graph``, compiled once against ``registry``."""

    return NodeDefinition.from_compiled_graphs(
        node_id, title=graph_title, description="d", category="test", versions={1: CompiledGraph.from_graph(record(*list(graph)), registry)},
    )
