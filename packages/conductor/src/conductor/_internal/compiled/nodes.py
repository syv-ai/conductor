"""The nodes of a compiled graph: a ``CompiledFunctionNode`` for a node whose definition is a function, and a
``CompiledSubgraphNode`` for a node whose definition is a subgraph. ``compiler.py`` makes them.

The example graph used in this file::

    lines(text: str) -> Series[str]            # called once; returns ["a", "b"]
    shout(line: str, mark: str = "!") -> str   # wired to lines.result: called once per line
    join(texts: Series[str]) -> str            # wired to shout.result: called once, with ["A!", "B!"]

``shout`` compiles to a ``CompiledFunctionNode`` with ``id="shout"``, ``loops=(OutputAddress("lines", "result"),)``,
an input plan for each of ``line`` (one item of ``lines.result``) and ``mark`` (the constant ``"!"``), and the
output ``result: str``. A compiled node has nothing from a run in it: it is made once and used by every run.
"""

from __future__ import annotations

import functools
import inspect
import typing
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from conductor._internal.address import split_address
from conductor._internal.ask import output_questions
from conductor._internal.compiled.input_plans import InputPlan, OutputAddress
from conductor._internal.field import Field, FieldDescription
from conductor._internal.hashing import short_hash
from conductor._internal.node_definition import FunctionDefinition, SubgraphDefinition
from conductor._internal.outcomes import Skipped, outputs_when_not_called, outputs_when_skipped
from conductor._internal.series import Series
from conductor._internal.value_type import Type

if TYPE_CHECKING:
    from conductor._internal.compiled.type import CompiledGraph


class RequiredBranch(NamedTuple):
    """One branch that must be chosen for a node to be called, or for an output to have a value.

    Example: ``route`` returns one of its outputs ``long`` and ``short`` per call, and ``shout`` is wired to
    ``route.long``. ``shout`` is called only where ``route`` chose ``long``:
    ``RequiredBranch("route", "length", "long", depth=0)``. Two edges into one input are allowed when what they
    read needs different outputs of one group, since a call then gets a value from at most one of them.

    Attributes:
        node_id: The id of the node that chooses, in the graph being compiled.
        group: The exclusive group of the outputs it chooses between.
        output: The output of the group that must be the chosen one.
        depth: How many loops the node that chooses is inside: a node inside the loop over ``lines.result``
            chooses once per line.
    """

    node_id: str
    group: str
    output: str
    depth: int


@dataclass(frozen=True, kw_only=True)
class _CompiledNodeBase:
    """What both kinds of node of a compiled graph have. Never made on its own.

    Attributes:
        id: The node's id in the graph record.
        loops: The loops the node is inside, outermost first, each named by the output it loops over.
        inputs: The input plan of each input, by name. Each plan's ``field`` is the input as compiled: it includes
            the inputs a ``compute_added_inputs`` hook added, and a type variable is replaced by the type its edges
            read.
        outputs: Each output, by name, as compiled: as the definition declares it, or as a ``compute_outputs``
            hook returned it.
        required_branches: The branches that must be chosen upstream for the node to be called.
    """

    id: str
    loops: tuple[OutputAddress, ...]
    inputs: dict[str, InputPlan]
    outputs: dict[str, Field]
    required_branches: frozenset[RequiredBranch]

    @functools.cached_property
    def upstream_ids(self) -> frozenset[str]:
        """The ids of the nodes that the edges into this node come from."""
        return frozenset(source.node_id for plan in self.inputs.values() for source in plan.sources)

    @functools.cached_property
    def plain_inputs(self) -> bool:
        """Whether no input's type can contain a mapping, so a call's arguments can be hashed without sorting
        keys first."""
        return all(plan.field.type.hashes_plainly for plan in self.inputs.values())

    def loops_around(self, output: str) -> tuple[OutputAddress, ...]:
        """Returns the loops ``output`` has one value per item of, outermost first: for a function node, the
        loops the node is inside. In the example, ``shout.loops_around("result")`` is the loop over
        ``lines.result``."""
        return self.loops

    def branches_required_for(self, output: str) -> frozenset[RequiredBranch]:
        """Returns the branches that must be chosen for ``output`` to have a value: those the node needs to be
        called, and ``output`` itself when it is in an exclusive group."""
        group = self.outputs[output].param.exclusive_group
        return self.required_branches | {RequiredBranch(self.id, group, output, len(self.loops))} if group else self.required_branches

    def described_outputs(self) -> list[FieldDescription]:
        """Returns each output as an editor shows it."""
        return [field.description for field in self.outputs.values()]


@typing.final
@dataclass(frozen=True, kw_only=True)
class CompiledFunctionNode(_CompiledNodeBase):
    """A node of a compiled graph whose definition is a function: ``shout`` in the example.

    Attributes:
        definition: The node definition, as its author declared it with ``@node``.
    """

    definition: FunctionDefinition[..., Any]

    @functools.cached_property
    def code_fingerprint(self) -> str:
        """A short hash of what a call's outputs depend on besides its arguments: the node's id and version, its
        code, the types of its inputs and outputs, and its loops.

        A run that reuses the outputs of an earlier run (``with_inputs(previous=run)``) reuses a call only while
        this is the same. So a node redeclared under the same id and version, but with other code, an input or
        output of another type, limit or choice, or inside other loops, is called again. The code is read from
        the function's own bytecode and constants, not from the functions it calls.
        """
        specs = (*(plan.field for plan in self.inputs.values()), *self.outputs.values())
        fields = [(spec.name, spec.whole.reads_as) for spec in specs]
        definition = self.definition
        # as text: the id of a call waiting for a person's answer and a saved run contain the hash, so no repr may change it
        loops = tuple([str(series) for series in self.loops])
        written = repr((definition.id, definition.version, _code_text(inspect.unwrap(definition.function).__code__), fields, loops))
        return short_hash(written.encode(), 8)

    @functools.cached_property
    def outputs_when_skipped(self) -> dict[str, Any]:
        """The outputs of the node where it was not called because it is on a branch that was not chosen:
        ``SKIPPED`` for each."""
        return outputs_when_skipped(self.outputs)

    @functools.cached_property
    def outputs_when_not_called(self) -> dict[str, Any]:
        """The outputs of the node where it was not called because an argument was ``None`` and its parameter
        does not accept ``None``."""
        return outputs_when_not_called(self.outputs)

    @functools.cached_property
    def skipped(self) -> Skipped:
        """The outcome of the node where it was not called because it is on a branch that was not chosen. Every
        such call of the node has this one object as its outcome."""
        return Skipped(self.outputs_when_skipped)

    @functools.cached_property
    def questions(self) -> dict[str, FieldDescription]:
        """The question a person is asked for each output when the function raises ``Ask``, made from the
        output's type; the ``Ask`` can add a prompt, choices and a default to it."""
        return output_questions(self.outputs)

    @property
    def functions_with_context(self) -> list[tuple[str, FunctionDefinition[..., Any]]]:
        """The node's id and definition when the function has a ``Context`` parameter; else an empty list."""
        return [(self.id, self.definition)] if self.definition.context else []


@typing.final
@dataclass(frozen=True, kw_only=True)
class CompiledSubgraphNode(_CompiledNodeBase):
    """A node of a compiled graph whose definition is a subgraph. The subgraph's own graph record is compiled
    on its own, as ``inner``, and the run calls the function nodes inside it in this node's loops.

    Its inputs are the inner graph's inputs (``inner.inputs``), by their names there, and its outputs are the inner
    graph's outputs (``inner.outputs``), by their addresses there. It has no policy (retries, threads, timeouts),
    and its outputs are never reused as a whole: each node inside is called by its own policy, and a later run
    reuses the outputs of the nodes inside one by one, as it would if they were drawn in the graph itself.

    An output is read outside as it would be if the subgraph's nodes were drawn in the outer graph. Example: the
    subgraph ``documents -> lines -> shout``, used as the node ``e``. ``lines`` is called once per document and
    ``shout`` once per line. So a node outside wired to ``e.shout.result`` that takes one ``str`` is called once
    per line of each document, and one that takes a ``Series[str]`` once per document. ``outputs`` has the output
    as ``shout`` declares it (``str``), and ``loops_around`` names the two loops. Its description says
    ``Series[Series[str]]``, one ``Series`` per loop.

    Attributes:
        definition: The node definition, as its author declared it with ``subgraph(...)``.
        inner: The subgraph's graph record, compiled on its own. It is never run on its own.
    """

    definition: SubgraphDefinition
    inner: CompiledGraph

    def loops_around(self, output: str) -> tuple[OutputAddress, ...]:
        """Returns the loops ``output`` has one value per item of, outermost first: the loops this node is
        inside, then the loops inside the subgraph.

        Example: ``e`` above is called once, and ``loops_around("shout.result")`` is
        ``(e.documents.result, e.lines.result)``. A loop inside is named by this node's id and the address of
        the list inside, so two outputs returned in the same loop inside name the same loop, and a node outside
        that reads both is called once per item with the two values of that item.
        """
        node_id, inner_output = split_address(output)
        inside = self.inner._nodes[node_id].loops_around(inner_output)
        return (*self.loops, *[OutputAddress(self.id, str(loop)) for loop in inside])

    def branches_required_for(self, output: str) -> frozenset[RequiredBranch]:
        """Returns the branches that must be chosen for ``output`` to have a value: those this node needs to be
        called, and those the output needs inside the subgraph.

        Example: the subgraph ``e`` is ``route -> shout`` and ``route -> whisper``, one on each output of
        ``route``. ``branches_required_for("shout.result")`` is ``RequiredBranch("e", "route.length",
        "route.long", 0)`` and that of ``whisper.result`` names ``route.short``, so one input outside may be
        wired to both, as it may with the same nodes drawn in the outer graph. A branch inside is named by this
        node's id, with the group and the output as addresses inside the subgraph.
        """
        node_id, inner_output = split_address(output)
        inside = self.inner._nodes[node_id].branches_required_for(inner_output)
        depth = len(self.loops)
        return self.required_branches | {
            RequiredBranch(self.id, f"{branch.node_id}.{branch.group}", f"{branch.node_id}.{branch.output}", depth + branch.depth)
            for branch in inside
        }

    @property
    def functions_with_context(self) -> list[tuple[str, FunctionDefinition[..., Any]]]:
        """The id and definition of each function node inside the subgraph, at any depth, with a ``Context``
        parameter."""
        return self.inner._needs_context

    def described_outputs(self) -> list[FieldDescription]:
        """Returns each output as an editor shows it: as a list with one level per loop inside the subgraph, with
        those loops by name.

        Example: ``shout.result`` of ``e`` above is described with ``is_series`` true, the item type
        ``Series[str]`` (so the whole output is ``Series[Series[str]]``), and the ``loops``
        ``["documents.result", "lines.result"]``.
        """
        described = []
        series_of: Any = Series  # typed Any: the type checker cannot follow ``Series[...]`` built at run time
        for name, field in self.outputs.items():
            loops = [str(loop).removeprefix(f"{self.id}.") for loop in self.loops_around(name)[len(self.loops) :]]
            if not loops:
                described.append(field.description)
                continue
            if field.is_series:  # an output that is itself a Series adds the innermost level
                loops.append(name)
            item: Any = field.type.hint
            for _ in loops[1:]:
                item = series_of[item]
            item_type = Type(item)
            series_field = field if field.is_series else Field(name, series_of[field.hint])
            described.append(series_field.description.model_copy(update={"type": item_type.id, "json_schema": item_type.json_schema, "loops": loops}))
        return described


#: One node of a compiled graph: a function, or a subgraph.
CompiledNode = CompiledFunctionNode | CompiledSubgraphNode


def _code_text(code: Any) -> str:
    """Returns a function's bytecode, constants and names as text that is the same in every process. A function
    defined inside it is written by its own code."""
    return f"{code.co_code.hex()}:{[_constant_text(c) for c in code.co_consts]}:{code.co_names}"


def _constant_text(constant: Any) -> str:
    """Returns a constant as text that is the same in every process. A set's members are sorted, because a set's
    order depends on the process's hash seed."""
    if hasattr(constant, "co_code"):
        return _code_text(constant)
    if isinstance(constant, frozenset):
        return f"frozenset({sorted(_constant_text(c) for c in constant)})"
    if isinstance(constant, tuple):
        return f"({', '.join(_constant_text(c) for c in constant)},)"
    return repr(constant)
