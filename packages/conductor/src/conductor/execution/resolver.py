"""Input resolution — what one node receives, read off ``CompiledGraph``."""

from collections.abc import Mapping
from typing import Any

from conductor._sentinel import is_skipped
from conductor.execution.results import extract_output
from conductor.graph.binding import Edges, Static
from conductor.graph.compiled import CompiledGraph
from conductor.ref import Ref
from conductor.series import Series
from conductor.types import NodeResult


class InputResolver:
    """Resolves a node's inputs from its bindings and the producers' results.

    Iterates the node's roster, never the binding table: a binding on a
    field the node does not have (``stale_binding``) is not a value
    anything receives. A ``Static`` is the author's typed value; an
    ``Edges`` is read off the producers' results with skipped values
    dropped — one series is received whole, a ``Series[X]`` input gathers
    what arrives into one series on the index compile named, otherwise
    exactly one value arrives. An input nothing binds is left to its
    declared default. A producer missing from ``results`` is the engine's
    bug and raises.
    """

    def __init__(self, compiled: CompiledGraph) -> None:
        self._compiled = compiled

    def resolve(self, node_id: str, results: Mapping[str, NodeResult]) -> dict[str, Any]:
        compiled = self._compiled
        node = compiled.node(node_id)
        statics = compiled.statics(node_id)
        inputs: dict[str, Any] = {}
        for inp in compiled.roster(node_id).inputs:
            binding = node.bindings.get(inp.name)
            if binding is None:
                continue
            if isinstance(binding, Static):
                inputs[inp.name] = statics[inp.name]
                continue
            values = _delivered(binding, results)
            if not values:
                continue
            if _is_series(inp.dtype):
                inputs[inp.name] = (
                    values[0]
                    if len(values) == 1 and isinstance(values[0], Series)
                    else Series(compiled.carried(Ref(node_id, inp.name)).index, values)
                )
            else:
                (inputs[inp.name],) = values
        return inputs


def _delivered(binding: Edges, results: Mapping[str, NodeResult]) -> list[Any]:
    """The values the edges deliver, in ref order, skipped ones dropped."""
    values: list[Any] = []
    for ref in binding.refs:
        result = results[ref.node_id]
        if is_skipped(result):
            continue
        value = extract_output(result, ref.field)
        if is_skipped(value):
            continue
        values.append(value)
    return values


def _is_series(dtype: Any) -> bool:
    return isinstance(dtype, type) and issubclass(dtype, Series)
