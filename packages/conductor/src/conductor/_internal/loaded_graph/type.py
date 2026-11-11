"""``LoadedGraph``: a stored graph checked against the nodes it uses, and its placed nodes.

A table is one value. An embedded graph is one node: its inputs are the inner
graph's inputs, its outputs the inner graph's outputs. It loads once, alone,
however many graphs embed it, and nothing of it is copied into the graph
around it. An author and a host see that one node, and an edge names its
output, never a node inside it. What a run walks of it is in ``tree.py``.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import inspect
import typing
from dataclasses import dataclass
from typing import Any

from conductor._internal.ask import output_questions
from conductor._internal.canonical_json import canonical_json
from conductor._internal.fields import Field, FieldDescription
from conductor._internal.graph_record import GraphRecord
from conductor._internal.loaded_graph.describe import (
    GraphDescription,
    GraphInput,
    GraphNodeDescription,
    PlacedFieldDescription,
    RunsOn,
    _feed,
    _runs_on,
)
from conductor._internal.loaded_graph.input_plans import Fed, InputPlan, Scope
from conductor._internal.node_definition import FunctionDefinition, GraphDefinition
from conductor._internal.problem import Problem
from conductor._internal.units import Skipped, outputs_not_called, outputs_not_taken


@dataclass(frozen=True, kw_only=True)
class _Placed:
    """What every node of a loaded graph has, whether it is a function or an embedded graph. Never made on its own.

    Attributes:
        id: The node's id in the record.
        scope: The scope it runs in.
        inputs: How each input gets its value, by name; the fields are the node's as its hooks made them.
        outputs: Each output's field, by name, as its hooks made them.
        opens: The ``Series`` outputs some reader works through one item at a time, which a run holds to
            ``max_items``. A reader placed after this node adds to it.
    """

    id: str
    scope: Scope
    inputs: dict[str, InputPlan]
    outputs: dict[str, Field]
    opens: set[str] = dataclasses.field(default_factory=set[str])

    @functools.cached_property
    def reads(self) -> frozenset[str]:
        """The nodes its inputs read."""
        return frozenset(source.node_id for plan in self.inputs.values() if isinstance(plan, Fed) for source in plan.sources)

    @functools.cached_property
    def plain_inputs(self) -> bool:
        """Whether no input can hold a mapping, so what a unit received needs no key sorting to hash."""
        return all(plan.spec.type.hashes_plainly for plan in self.inputs.values())


@typing.final
@dataclass(frozen=True, kw_only=True)
class PlacedFunction(_Placed):
    """A node of a loaded graph that is a function.

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
        specs = (*(plan.spec for plan in self.inputs.values()), *self.outputs.values())
        fields = [(spec.name, spec.whole.reads_as) for spec in specs]
        definition = self.definition
        written = repr((definition.id, definition.version, _code_text(inspect.unwrap(definition.function).__code__), fields, self.scope))
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
class PlacedGraph(_Placed):
    """A node of a loaded graph that is an embedded graph: the nodes of ``inner`` run where it stands.

    Its inputs are the inner graph's (``inner.inputs``) by their names there, and its outputs the inner graph's
    (``inner.outputs``) by their addresses there. It has no policy and no unit of it is reused whole: the nodes
    inside are called by their own policies, and a later run takes their earlier work node by node, as it would
    with the same nodes drawn in the graph itself.

    Attributes:
        definition: The node as its author declared it.
        inner: The embedded graph, loaded.
    """

    definition: GraphDefinition
    inner: LoadedGraph


#: One node of a loaded graph: a function, or an embedded graph.
PlacedNode = PlacedFunction | PlacedGraph


@dataclass(frozen=True, slots=True)
class InputField:
    """What a run keeps of a graph input.

    Attributes:
        name: Its name in ``graph.inputs``: the input's own, or its address where two inputs share a name.
        address: Where the caller's value is kept, ``node.input``.
        spec: Its field, whose default is what it holds when the caller gives nothing: the record's default, else
            the node's, and ``REQUIRED`` when the caller must give one.
    """

    name: str
    address: str
    spec: Field


@dataclass(frozen=True, eq=False)
class LoadedGraph:
    """A stored graph checked against the nodes it uses: what loading makes, an embedded graph's node holds and a
    run reads. The ``Graph`` a host holds is one, with the verbs that start a run.

    ``problems`` is everything wrong with it, and a graph with none runs.

    Attributes:
        record: The stored graph it was loaded from.
        problems: Everything wrong with it; empty for a graph that runs.
        inputs: The open inputs of its first nodes, by name (``node.input`` where two share a name): what a
            caller fills.
    """

    record: GraphRecord
    problems: list[Problem]
    inputs: dict[str, GraphInput]
    #: Each node whose fields are known, by id, each after the nodes its inputs read. A graph with problems holds
    #: the nodes they are on too, with ``Unset`` for an input that gets nothing, and never runs.
    _nodes: dict[str, PlacedNode]
    #: Each graph input as a run keeps it, by address.
    _input_fields: dict[str, InputField]
    #: The nodes that read each node.
    _readers: dict[str, list[PlacedNode]]
    #: The nodes with a ``FromRun`` parameter, an embedded graph's too, as ``from_run_values`` reads them.
    _needs_from_run: list[tuple[str, FunctionDefinition[..., Any]]]

    def __repr__(self) -> str:
        state = "runnable" if self.is_runnable else f"{len(self.problems)} problem{'s' if len(self.problems) != 1 else ''}"
        return f"<Graph of {len(self.record.nodes)} node{'s' if len(self.record.nodes) != 1 else ''}, {state}>"

    # -- what a host reads ------------------------------------------------------

    @property
    def is_runnable(self) -> bool:
        """Whether the graph has no problems, so it runs."""
        return not self.problems

    @functools.cached_property
    def outputs(self) -> list[str]:
        """The outputs of the nodes nothing reads, as addresses."""
        return [f"{p.id}.{o}" for p in self._nodes.values() if not self._readers[p.id] for o in p.outputs]

    @property
    def plan(self) -> dict[str, RunsOn]:
        """Where each node runs: ``once``, or ``per_item`` of the innermost Series of its scope."""
        return {placed.id: _runs_on(placed.scope) for placed in self._nodes.values()}

    @property
    def exclusive_groups(self) -> dict[str, dict[str, list[str]]]:
        """Each node's exclusive groups and their outputs, for a host to say which way a run went."""
        found: dict[str, dict[str, list[str]]] = {}
        for placed in self._nodes.values():
            for name, spec in placed.outputs.items():
                if spec.param.exclusive_group:
                    found.setdefault(placed.id, {}).setdefault(spec.param.exclusive_group, []).append(name)
        return found

    def describe(self) -> GraphDescription:
        """The loaded graph as an editor draws it: every node of the record, with its fields and where it runs.

        Each node lists its own ``problems``, and ``problems`` beside the nodes lists every one. A node with a
        problem is drawn with its fields all the same, an input whose edge is refused fed ``unset``, and so is a
        node that reads it. A ``type_mismatch`` names the input in its ``loc``; a ``cycle`` and
        ``series_not_nested`` name the node. Only a node whose fields aren't known is there with ``loaded``
        false and no fields, its problems saying why: an unknown node or version, an id no address can say, a
        graph that embeds itself, a hook that failed or wasn't called for a value it reads, an embedded graph
        with problems of its own. An embedded graph is one node, with the fields it offers; one with problems
        isn't placed, and its problems are this graph's, each ``loc`` the embedding node's id and then the
        place inside it.
        """
        nodes: dict[str, GraphNodeDescription] = {}
        for record in self.record.nodes:
            own = [p for p in self.problems if p.loc[:1] == (record.id,)]
            placed = self._nodes.get(record.id)
            if placed is None:
                nodes[record.id] = GraphNodeDescription(node=record.node, version=record.version, loaded=False, problems=own)
                continue
            nodes[record.id] = GraphNodeDescription(
                node=record.node, version=record.version, loaded=True, problems=own, runs=_runs_on(placed.scope),
                inputs=[PlacedFieldDescription(field=plan.spec.description, feed=_feed(placed.scope, plan)) for plan in placed.inputs.values()],
                outputs=[spec.description for spec in placed.outputs.values()],
            )
        return GraphDescription(nodes=nodes, inputs=dict(self.inputs), outputs=self.outputs, problems=list(self.problems))

    @functools.cached_property
    def fingerprint(self) -> str:
        """A short hash of what a run of this graph rests on: the record, but for each node's ``display``, which
        conductor never reads, and the fingerprint of every graph embedded in it.

        Keys are sorted first, so a record a JSON store gives back in another key order has the same fingerprint.
        A saved run names it, and loading a save into a graph whose fingerprint differs is refused.
        """
        own = self.record.model_dump(mode="json", exclude={"nodes": {"__all__": {"display"}}})
        embedded = {p.id: p.inner.fingerprint for p in self._nodes.values() if isinstance(p, PlacedGraph)}
        return hashlib.blake2b(canonical_json([own, embedded]), digest_size=8).hexdigest()


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
