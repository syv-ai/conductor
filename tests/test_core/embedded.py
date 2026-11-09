"""A test helper: a stored graph offered as a node, the way a host offers one."""


from conductor import CompiledGraph, Graph, GraphNode, NodeDefinition, NodeRegistry


def embedded_graph_node(node_id: str, graph: tuple[GraphNode, ...], registry: NodeRegistry, graph_title: str = "Embedded") -> NodeDefinition:
    """A definition whose one version is ``graph``, compiled once against ``registry``."""

    return NodeDefinition.from_compiled_graphs(
        node_id, title=graph_title, description="d", category="test", versions={1: CompiledGraph.from_graph(Graph(nodes=list(graph)), registry)},
    )
