"""Skip propagation."""

from collections.abc import Mapping
from typing import Any

from conductor._sentinel import is_skipped
from conductor.execution.results import extract_output
from conductor.graph.binding import Edges
from conductor.graph.compiled import CompiledGraph
from conductor.ref import Ref


def should_skip_node(compiled: CompiledGraph, node_id: str, results: Mapping[str, Any]) -> bool:
    """Is every value this node's edges deliver ``SKIPPED``?

    A node with no edges never skips. Read off the node's interface, so a stale
    binding delivers nothing. A producer missing from ``results`` is the
    engine's bug and raises.
    """
    delivered = False
    for inp in compiled.node(node_id).interface.inputs:
        binding = compiled.field(Ref(node_id, inp.name)).binding
        if not isinstance(binding, Edges):
            continue
        for ref in binding.refs:
            delivered = True
            result = results[ref.node_id]
            if is_skipped(result):
                continue
            if not is_skipped(extract_output(result, ref.field)):
                return False
    return delivered
