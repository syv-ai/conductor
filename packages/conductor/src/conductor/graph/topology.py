"""An execution order from the dependency map, and the wire maps the engine reads."""

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping

from conductor.errors import CycleDetectionError
from conductor.graph.binding import Sources
from conductor.graph.model import GraphNode


def topological_sort(dependencies: Mapping[str, frozenset[str]]) -> list[str]:
    """Node ids in an order where every dependency precedes its dependent.

    Kahn's algorithm over ``dependencies_of``'s map. Raises
    ``CycleDetectionError`` if the graph contains cycles. A dependency on
    an id the map does not hold is ignored here; compile reports it on
    the binding that names it.
    """
    ids = set(dependencies)
    in_degree = {i: len(deps & ids) for i, deps in dependencies.items()}
    dependents: dict[str, list[str]] = defaultdict(list)
    for i, deps in dependencies.items():
        for dep in deps & ids:
            dependents[dep].append(i)

    queue = deque(i for i, degree in in_degree.items() if degree == 0)
    result: list[str] = []
    while queue:
        i = queue.popleft()
        result.append(i)
        for dependent in dependents.get(i, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(ids):
        raise CycleDetectionError(f"Cycle detected involving nodes: {ids - set(result)}")

    return result


def _wires(nodes: Iterable[GraphNode]) -> Iterable[tuple[str, str, str, str]]:
    """Every ``(target_id, target_handle, source_id, source_handle)`` the bindings hold, in ref order."""
    for node in nodes:
        for handle, binding in node.bindings.items():
            if isinstance(binding, Sources):
                for ref in binding.refs:
                    yield node.id, handle, ref.node_id, ref.field


def build_edge_map(
    nodes: Iterable[GraphNode],
) -> dict[tuple[str, str], list[tuple[str, str, str]]]:
    """Build ``(target_id, target_handle) -> [(source_id, source_handle, wire_id), ...]``.

    The wire id names the wire for the resolver's skip bookkeeping; it is
    ``"source.handle->target.handle"``, derived and stored nowhere.
    """
    edge_map: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    for target_id, target_handle, source_id, source_handle in _wires(nodes):
        edge_map[(target_id, target_handle)].append(
            (source_id, source_handle, f"{source_id}.{source_handle}->{target_id}.{target_handle}")
        )
    return dict(edge_map)


def build_incoming_map(
    nodes: Iterable[GraphNode],
) -> dict[str, list[tuple[str, str, str, str]]]:
    """Build ``target_id -> [(target_handle, source_id, source_handle, wire_id), ...]``.

    The inverted view of ``build_edge_map``; ``should_skip_node`` and
    ``InputResolver.resolve`` read it per node.
    """
    incoming: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for target_id, target_handle, source_id, source_handle in _wires(nodes):
        incoming[target_id].append(
            (target_handle, source_id, source_handle, f"{source_id}.{source_handle}->{target_id}.{target_handle}")
        )
    return dict(incoming)
