"""The nodes of a compiled graph, as the compiler places them: a function node, or a stored graph used as a node."""

from __future__ import annotations

import functools
import hashlib
import inspect
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from conductor._internal.ask import output_questions
from conductor._internal.compiled.input_plans import (
    InputFromEdges,
    InputPlan,
    OutputAddress,
)
from conductor._internal.field import Field, FieldDescription
from conductor._internal.node_definition import FunctionDefinition, StoredGraphDefinition
from conductor._internal.outcomes import Skipped, outputs_when_not_called, outputs_when_skipped
from conductor._internal.series import Series
from conductor._internal.value_type import Type

if TYPE_CHECKING:
    from conductor._internal.compiled.type import CompiledGraph


@dataclass(frozen=True, kw_only=True)
class _CompiledNodeBase:
    """What every node of a compiled graph has, whether it is a function or a subgraph. Never made on its own.

    Attributes:
        id: The node's id in the record.
        loops: The scope it runs in.
        inputs: How each input gets its value, by name; the fields are the node's as its hooks made them.
        outputs: Each output's field, by name, as its hooks made them.
    """

    id: str
    loops: tuple[OutputAddress, ...]
    inputs: dict[str, InputPlan]
    outputs: dict[str, Field]

    @functools.cached_property
    def reads(self) -> frozenset[str]:
        """The nodes its inputs read."""
        return frozenset(source.node_id for plan in self.inputs.values() if isinstance(plan, InputFromEdges) for source in plan.sources)

    @functools.cached_property
    def plain_inputs(self) -> bool:
        """Whether no input can hold a mapping, so what a unit received needs no key sorting to hash."""
        return all(plan.field.type.hashes_plainly for plan in self.inputs.values())

    def loops_around(self, output: str) -> tuple[OutputAddress, ...]:
        """The lists looped over to make one value of ``output``, outermost first: the loops the node is called in."""
        return self.loops

    def described_outputs(self) -> list[FieldDescription]:
        """Each output as an editor reads it."""
        return [field.description for field in self.outputs.values()]


@typing.final
@dataclass(frozen=True, kw_only=True)
class CompiledFunctionNode(_CompiledNodeBase):
    """A node of a compiled graph that is a function.

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
        loops = tuple([str(series) for series in self.loops])  # as addresses: a waiting id and a save hold the digest, so no class's repr is in it
        written = repr((definition.id, definition.version, _code_text(inspect.unwrap(definition.function).__code__), fields, loops))
        return hashlib.blake2b(written.encode(), digest_size=8).hexdigest()

    @functools.cached_property
    def outputs_when_skipped(self) -> dict[str, Any]:
        """What a unit of the node holds where the node wasn't called for a branch not taken."""
        return outputs_when_skipped(self.outputs)

    @functools.cached_property
    def outputs_when_not_called(self) -> dict[str, Any]:
        """What a unit of the node holds where the node wasn't called for a blank."""
        return outputs_when_not_called(self.outputs)

    @functools.cached_property
    def skipped(self) -> Skipped:
        """The unit the node comes to where it wasn't called for a branch not taken; every such unit of it is this one."""
        return Skipped(self.outputs_when_skipped)

    @functools.cached_property
    def questions(self) -> dict[str, FieldDescription]:
        """What a person is asked for each output when the node raises ``Ask``, before the ``Ask`` adds to it."""
        return output_questions(self.outputs)


@typing.final
@dataclass(frozen=True, kw_only=True)
class CompiledSubgraphNode(_CompiledNodeBase):
    """A node of a compiled graph that is a stored graph: the compiler makes one where a record uses a
    ``StoredGraphDefinition``, checking that definition's record alone as ``inner``, whose nodes run where it stands.

    Its inputs are the inner graph's (``inner.inputs``) by their names there, and its outputs the inner graph's
    (``inner.outputs``) by their addresses there. It has no policy and no unit of it is reused whole: the nodes
    inside are called by their own policies, and a later run takes their earlier work node by node, as it would
    with the same nodes drawn in the graph itself.

    An output is read outside as it would be with the same nodes drawn in the outer graph. Example: the stored
    graph ``documents -> lines -> shout``, used as the node ``e``. ``lines`` is called once per document and
    ``shout`` once per line, so a node outside wired to ``e.shout.result`` that takes one ``str`` is called once
    per line of each document, and one that takes a ``Series[str]`` once per document. ``outputs`` holds the
    field as ``shout`` declares it (``str``), and ``loops_around`` names the two loops. The description says
    ``Series[Series[str]]``, one ``Series`` per loop.

    Attributes:
        definition: The node as its author declared it.
        inner: The subgraph, checked alone; it is never run alone.
    """

    definition: StoredGraphDefinition
    inner: CompiledGraph

    def loops_around(self, output: str) -> tuple[OutputAddress, ...]:
        """The lists looped over to make one value of ``output``, outermost first: the loops this node is called
        in, then the loops inside the stored graph.

        Example: ``e`` above is called once, and ``loops_around("shout.result")`` is
        ``(e.documents.result, e.lines.result)``. A loop inside is named by this node's id and the address of
        the list inside, so two outputs returned in the same loop inside name the same loop, and a node outside
        that reads both is called once per item with the two values of that item.
        """
        node_id, _, inner_output = output.partition(".")
        inside = self.inner._nodes[node_id].loops_around(inner_output)
        return (*self.loops, *[OutputAddress(self.id, str(loop)) for loop in inside])

    def described_outputs(self) -> list[FieldDescription]:
        """Each output as an editor reads it: one ``Series`` for each loop inside, and the loops by name.

        Example: ``shout.result`` of ``e`` above is ``many`` with the type ``Series[str]``, which reads
        ``Series[Series[str]]``, and its ``loops`` are ``["documents.result", "lines.result"]``.
        """
        described = []
        series_of: Any = Series  # a hint built from a value, which a type checker reads no further
        for name, field in self.outputs.items():
            loops = [str(loop).removeprefix(f"{self.id}.") for loop in self.loops_around(name)[len(self.loops) :]]
            if not loops:
                described.append(field.description)
                continue
            if field.many:  # the node's own Series is the innermost one
                loops.append(name)
            item: Any = field.type.hint
            for _ in loops[1:]:
                item = series_of[item]
            item_type = Type(item)
            series_field = field if field.many else Field(name, series_of[field.hint])
            described.append(series_field.description.model_copy(update={"type": item_type.id, "json_schema": item_type.json_schema, "loops": loops}))
        return described


#: One node of a compiled graph: a function, or a subgraph.
CompiledNode = CompiledFunctionNode | CompiledSubgraphNode


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
