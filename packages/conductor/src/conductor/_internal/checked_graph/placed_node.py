"""The nodes of a checked graph, as the loader places them: a function node, or a stored graph used as a node."""

from __future__ import annotations

import functools
import hashlib
import inspect
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from conductor._internal.ask import output_questions
from conductor._internal.checked_graph.input_plans import InputFedByEdges, InputPlan, Scope
from conductor._internal.fields import Field, FieldDescription
from conductor._internal.node_definition import FunctionDefinition, StoredGraphDefinition
from conductor._internal.units import Skipped, outputs_not_called, outputs_not_taken

if TYPE_CHECKING:
    from conductor._internal.checked_graph.type import CheckedGraph


@dataclass(frozen=True, kw_only=True)
class _Placed:
    """What every node of a checked graph has, whether it is a function or an embedded graph. Never made on its own.

    Attributes:
        id: The node's id in the record.
        scope: The scope it runs in.
        inputs: How each input gets its value, by name; the fields are the node's as its hooks made them.
        outputs: Each output's field, by name, as its hooks made them.
    """

    id: str
    scope: Scope
    inputs: dict[str, InputPlan]
    outputs: dict[str, Field]

    @functools.cached_property
    def reads(self) -> frozenset[str]:
        """The nodes its inputs read."""
        return frozenset(source.node_id for plan in self.inputs.values() if isinstance(plan, InputFedByEdges) for source in plan.sources)

    @functools.cached_property
    def plain_inputs(self) -> bool:
        """Whether no input can hold a mapping, so what a unit received needs no key sorting to hash."""
        return all(plan.field.type.hashes_plainly for plan in self.inputs.values())


@typing.final
@dataclass(frozen=True, kw_only=True)
class PlacedFunction(_Placed):
    """A node of a checked graph that is a function.

    Attributes:
        definition: The node as its author declared it.
    """

    definition: FunctionDefinition[..., Any]

    @functools.cached_property
    def reuse_digest(self) -> str:
        """What a unit's result rests on besides what it received: the node's version, its code, its fields and its scope.

        So a node redeclared under the same id and version, with other code, a field of another type, limit or
        choice, or one that runs in another scope, is called again by ``run.again``. Code is read from the function's own bytecode and
        constants, not from what it calls.
        """
        specs = (*(plan.field for plan in self.inputs.values()), *self.outputs.values())
        fields = [(spec.name, spec.whole.reads_as) for spec in specs]
        definition = self.definition
        scope = tuple([str(series) for series in self.scope])  # as addresses: a waiting id and a save hold the digest, so no class's repr is in it
        written = repr((definition.id, definition.version, _code_text(inspect.unwrap(definition.function).__code__), fields, scope))
        return hashlib.blake2b(written.encode(), digest_size=8).hexdigest()

    @functools.cached_property
    def outputs_not_taken(self) -> dict[str, Any]:
        """What a unit of the node holds where the node wasn't called for a branch not taken."""
        return outputs_not_taken(self.outputs)

    @functools.cached_property
    def outputs_not_called(self) -> dict[str, Any]:
        """What a unit of the node holds where the node wasn't called for a blank."""
        return outputs_not_called(self.outputs)

    @functools.cached_property
    def skipped(self) -> Skipped:
        """The unit the node comes to where it wasn't called for a branch not taken; every such unit of it is this one."""
        return Skipped(self.outputs_not_taken)

    @functools.cached_property
    def questions(self) -> dict[str, FieldDescription]:
        """What a person is asked for each output when the node raises ``Ask``, before the ``Ask`` adds to it."""
        return output_questions(self.outputs)


@typing.final
@dataclass(frozen=True, kw_only=True)
class PlacedEmbeddedGraph(_Placed):
    """A node of a checked graph that is a stored graph: the loader makes one where a record uses a
    ``StoredGraphDefinition``, checking that definition's record alone as ``inner``, whose nodes run where it stands.

    Its inputs are the inner graph's (``inner.inputs``) by their names there, and its outputs the inner graph's
    (``inner.outputs``) by their addresses there. It has no policy and no unit of it is reused whole: the nodes
    inside are called by their own policies, and a later run takes their earlier work node by node, as it would
    with the same nodes drawn in the graph itself.

    Attributes:
        definition: The node as its author declared it.
        inner: The embedded graph, checked alone; it is never run alone.
    """

    definition: StoredGraphDefinition
    inner: CheckedGraph


#: One node of a checked graph: a function, or an embedded graph.
PlacedNode = PlacedFunction | PlacedEmbeddedGraph


def _code_text(code: Any) -> str:
    """A function's bytecode and constants as text that is the same in every process: a nested function by its own code."""
    return f"{code.co_code.hex()}:{[_constant_text(c) for c in code.co_consts]}:{code.co_names}"


def _constant_text(constant: Any) -> str:
    """A constant as text that is the same in every process: a set's members sorted, as its order follows the hash seed."""
    if hasattr(constant, "co_code"):
        return _code_text(constant)
    if isinstance(constant, frozenset):
        return f"frozenset({sorted(_constant_text(c) for c in constant)})"
    if isinstance(constant, tuple):
        return f"({', '.join(_constant_text(c) for c in constant)},)"
    return repr(constant)
