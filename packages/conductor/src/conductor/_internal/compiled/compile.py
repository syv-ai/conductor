"""Compiling a graph record: check every node and edge against the node definitions, and work out once, for
every run, where each input gets its value, which loops each node is inside, and the order of the nodes.

The example graph used in this file::

    lines(text: str) -> Series[str]            # called once; returns ["a", "b"]
    shout(line: str, mark: str = "!") -> str   # wired to lines.result: called once per line
    join(texts: Series[str]) -> str            # wired to shout.result: called once, with ["A!", "B!"]

``compile_graph`` puts the nodes in the order ``lines``, ``shout``, ``join`` and gives every input a compiled input
(``input.py``): ``lines.text`` is a graph input, ``shout.mark`` is the constant ``"!"``, ``shout.line`` reads
one item of ``lines.result``, and ``join.texts`` collects what ``shout`` returned for every line. ``shout`` is
inside the loop over ``lines.result``; ``lines`` and ``join`` are inside no loop.

Compiling never raises for a wrong graph. What is wrong becomes a ``Problem`` in the graph's ``problems``, naming
the node and the input it is on, and a graph with problems does not run.

The declared types alone decide the loops:

- An input that takes one ``T``, wired to an output declared ``Series[T]``, reads one item per call: its node is
  inside the loop over that output. A ``Table`` input wired to a ``Series`` of tables reads one table per call.
- An input wired to an output of a node outside the loop gets the same value on every call inside it: an outer
  value.
- An input declared ``Series[T]``, or ``Table[R]`` wired to records, collects. Wired to an output inside a loop, it
  gets one value per item of the innermost of those loops, as a list, and its node is outside that loop. Wired to
  an output inside no loop, it gets that one value as a list of one.
- A node is inside the loops of its input with the most loops. Every other input's loops must be the outermost of
  those, because the items of two different lists are never paired; an input whose loops are not has the problem
  ``series_not_nested``.
- A loop is named by the output it loops over, so every node that reads items of ``lines.result`` is inside the
  same loop.
"""

from __future__ import annotations

import heapq
import logging
import typing
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, TypeVar

from pydantic import MISSING
from typing_extensions import is_protocol

from conductor._internal.base import REQUIRED, Address
from conductor._internal.context import ContextParameter, what_the_context_lacks
from conductor._internal.field import Field
from conductor._internal.gate import (
    RefusedError,
    bound_field,
    read_constant,
    read_hook_fields,
    read_record,
)
from conductor._internal.node import (
    FunctionDefinition,
    HookName,
    NodeDefinition,
    SubgraphDefinition,
    read_by_hooks,
)
from conductor._internal.problem import NodeError, Problem
from conductor._internal.problem import catalogue as problem
from conductor._internal.record import (
    GraphRecord,
    NodeRecord,
    RecordConstant,
    RecordEdges,
    RecordGraphInput,
)
from conductor._internal.types import Type

from .input import (
    CollectingInput,
    ConstantInput,
    Edge,
    EdgeInput,
    GraphInput,
    Input,
    MissingInput,
)
from .node import (
    CompiledFunctionNode,
    CompiledNode,
    CompiledSubgraphNode,
    RequiredBranch,
)
from .type import CompiledGraph

_logger = logging.getLogger("conductor")


class Definitions(Protocol):
    """The node definitions a graph is compiled against: a ``NodeRegistry``. ``definitions["shout"]`` is the newest
    version of ``shout``, and a ``KeyError`` for an id it does not have; iterating gives every definition."""

    def __getitem__(self, node_id: str, /) -> NodeDefinition: ...

    def __iter__(self) -> Iterator[NodeDefinition]: ...


def compile_graph[GraphT: CompiledGraph](
    record: GraphRecord | str | bytes, definitions: Definitions, context_type: type | None, graph_type: type[GraphT],
) -> GraphT:
    """Compiles ``record`` and returns it as a ``graph_type``. Every compiled graph a host gets is made here.

    Args:
        record: The graph record, or its JSON text.
        definitions: The registry: the newest version of each node definition, looked up by id. An older version a
            node asks for is found through it.
        context_type: The class of the run's context, as the registry names it; ``None`` when it names none.
        graph_type: The class of the graph to return: ``Graph``, the class a host uses.

    Never raises for a wrong graph: the returned graph's ``problems`` say what is wrong.

    Raises:
        RefusedError: ``record`` is not a graph record's shape, as ``read_record`` raises.
    """
    return _Compiler(definitions, context_type, (), {}).compile(record, graph_type)



#: Stands in for a constant or default of the graph record that its input's type rejected, while a node is
#: compiled. The input gets ``MissingInput``, and the problem ``invalid_value`` says why.
_INVALID: Any = object()


@dataclass(frozen=True, slots=True)
class _InputPlans:
    """The compiled inputs of one node, and what follows from them.

    Attributes:
        inputs: The compiled input of each input, by name.
        loops: The loops the node is inside, outermost first. In the example, ``shout`` is inside the loop over
            ``lines.result``.
        required_branches: The branches that must be chosen upstream for the node to be called.
        bound: For a node whose input types have type variables (``T``), the type each variable gets from the
            edges into the node. ``None`` for every other node, which is nearly all of them.
    """

    inputs: dict[str, Input]
    loops: tuple[Address, ...]
    required_branches: frozenset[RequiredBranch]
    bound: Mapping[TypeVar, Any] | None


class _Compiler:
    """Compiles one graph record. It compiles the nodes one at a time, each after the nodes upstream of it, and
    keeps what it has found so far. A subgraph used as a node is compiled by a ``_Compiler`` of its own.

    Attributes:
        definitions: The registry: the newest version of each node definition, looked up by id.
        context_type: The class of the run's context, which each node's ``Context[X]`` is checked against; ``None``
            when the registry names none.
        inside_subgraphs: The ids of the subgraph definitions whose records contain this record, outermost first.
            A subgraph found among them contains itself.
        compiled_subgraphs: Each subgraph compiled so far, by its id, its version and the subgraphs it is inside.
            Every compiler of one ``compile_graph`` call shares it, so a subgraph used twice is compiled once.
        problems: The problems found so far.
        nodes: Each node compiled so far, by id.
        graph_inputs: Each graph input found so far, by name, in the order found: the field the caller gives a
            value for, with the type of the inputs it feeds and the default the caller's value replaces.
        record: The graph record being compiled.
        position: Each node's index among the record's node ids, sorted: it breaks ties in the order of the nodes.
    """

    def __init__(
        self, definitions: Definitions, context_type: type | None, inside_subgraphs: tuple[str, ...],
        compiled_subgraphs: dict[tuple[str, int, tuple[str, ...]], CompiledGraph],
    ) -> None:
        self.definitions, self.context_type = definitions, context_type
        self.inside_subgraphs, self.compiled_subgraphs = inside_subgraphs, compiled_subgraphs
        self.problems: list[Problem] = []
        self.nodes: dict[str, CompiledNode] = {}
        self.graph_inputs: dict[str, Field] = {}
        self.record = GraphRecord()
        self.position: dict[str, int] = {}

    def compile[GraphT: CompiledGraph](self, record: GraphRecord | str | bytes, graph_type: type[GraphT]) -> GraphT:
        """Returns ``record`` compiled, as a ``graph_type``.

        It never raises for a wrong record. The graph has the problems, and every node whose inputs and outputs
        could be worked out. A node whose definition is not found, or whose hook failed, is left out.
        """
        self.record = read_record(record)
        ids = sorted(self.record.nodes)  # the record keeps no order of its nodes, so ties go by id
        self.position = {node_id: i for i, node_id in enumerate(ids)}
        edges_into = {node_id: _edges_into(self.record.nodes[node_id]) for node_id in ids}
        for node_id in self._order(ids, edges_into):
            node_record = self.record.nodes[node_id]
            match self._resolve(node_id, node_record):
                case FunctionDefinition() as function:
                    self._compile_function_node(node_id, node_record, function, edges_into[node_id])
                case SubgraphDefinition() as subgraph:
                    self._compile_subgraph_node(node_id, node_record, subgraph, edges_into[node_id])
                case None:
                    pass
        outputs: dict[str, Edge] = {}
        for name, address in sorted(self.record.outputs.items()):
            source = self._sources("outputs", name, (address,))
            if source is not None:
                outputs[name] = source[0]
        return graph_type(self.record, self.problems, self.nodes, self.graph_inputs, outputs, self.context_type)

    def _resolve(self, node_id: str, record: NodeRecord) -> NodeDefinition | None:
        """Returns the node definition a node of the record names, at the record's version; ``None``, with a
        problem, when there is none."""
        try:
            newest = self.definitions[record.node]
        except KeyError:
            self.problems.append(problem.unknown_node(node_id, node=record.node, choices=sorted(definition.id for definition in self.definitions)))
            return None
        if record.version not in newest.versions:
            self.problems.append(problem.unknown_version(node_id, node=record.node, version=record.version, newest=newest.version))
            return None
        return newest.version_of(record.version)

    def _order(self, order: list[str], edges_into: Mapping[str, Mapping[str, tuple[Address, ...]]]) -> list[str]:
        """Returns the node ids in the order they are compiled: each after the nodes its edges come from, and
        by id where that leaves a choice.

        Every node on a cycle has the problem ``cycle``. To go on, the cycle's first node by id is put
        first. The edge into it from inside the cycle then comes from a node that is not compiled yet, so the
        input it leads to gets no value. The rest of the cycle follows in order.
        """
        position = self.position
        reads = {n: frozenset({source.node for sources in edges_into[n].values() for source in sources} & position.keys()) for n in order}
        downstream: dict[str, list[str]] = {n: [] for n in order}
        for n, sources in reads.items():
            for source in sources:
                downstream[source].append(n)
        waiting = {n: len(sources) for n, sources in reads.items()}
        ready = [position[n] for n in order if not waiting[n]]
        heapq.heapify(ready)
        done: list[str] = []
        on_cycles: list[str] | None = None
        while len(done) < len(order):
            if not ready:  # every node not yet ordered is on a cycle or downstream of one
                if on_cycles is None:
                    stuck = {n for n in order if waiting[n]}
                    on_cycles = [n for n in order if n in stuck and _reaches(n, n, downstream, stuck)]  # on a cycle, not only downstream of one
                    self.problems += [problem.cycle(n) for n in on_cycles]
                first = next(n for n in on_cycles if waiting[n] > 0)
                waiting[first] = 0
                ready.append(position[first])
            n = order[heapq.heappop(ready)]
            done.append(n)
            for reader in downstream[n]:
                waiting[reader] -= 1
                if not waiting[reader]:
                    heapq.heappush(ready, position[reader])
        return done

    # -- compiling a node -------------------------------------------------------

    def _compile_function_node(
        self, node_id: str, record: NodeRecord, definition: FunctionDefinition[..., Any], edges: Mapping[str, tuple[Address, ...]],
    ) -> None:
        """Compiles a node whose definition is a function, and adds it to ``nodes``.

        Its inputs are those its definition declares and those its ``compute_added_inputs`` hook adds; each gets
        a compiled input. Its outputs are those its definition declares, or what its ``compute_outputs`` hook
        returns, given the types its edges read. When a hook fails, the node is left out and the graph has the
        problem. Every other node is added, even when some of its inputs have problems.
        """
        if definition.context is not None:
            self._check_context(node_id, definition.context)
        fields = dict(definition.inputs)
        constants = self._constants_of(node_id, record, fields)
        added: dict[str, Field] = {}
        if definition.compute_added_inputs is not None:
            made = self._hook(node_id, definition, "compute_added_inputs", definition.compute_added_inputs, constants, {}, ())
            if made is None:
                return
            added = made
            taken = sorted(added.keys() & (fields.keys() | ({definition.context.name} if definition.context is not None else set())))
            if taken:
                self.problems.append(problem.compute_hook_failed(node_id, reason=f"it adds {taken[0]!r}, which the node already declares", error="added_input_taken"))
                return
            fields.update(added)
            constants.update(self._constants_of(node_id, record, added))
        planned = self._plan_inputs(node_id, record, fields, constants, read_by_hooks(definition), edges, generic=definition.generic)
        bound = planned.bound
        outputs: dict[str, Field] | None = dict(definition.outputs)
        if bound is not None:
            outputs, refused = _bind(node_id, definition.outputs, bound)
            self.problems += [found for problems in refused.values() for found in problems]
        if definition.compute_outputs is not None:
            arriving = {name: node_input.sources[0].type.bare for name, node_input in planned.inputs.items() if node_input.sources}
            outputs = self._hook(node_id, definition, "compute_outputs", definition.compute_outputs, constants, arriving, tuple(added))
        if outputs is not None:
            self.nodes[node_id] = CompiledFunctionNode(
                id=node_id, loops=planned.loops, inputs=planned.inputs, outputs=outputs, required_branches=planned.required_branches,
                definition=definition,
            )

    def _check_context(self, node_id: str, context: ContextParameter) -> None:
        """Adds a problem when the registry's context class is not what the node's ``Context[X]`` asks for.

        Example: ``sign(ctx: Context[HostContext], text: str)`` fits a registry that names ``HostContext`` or a
        subclass of it. ``fetch(ctx: Context[HasHttpClient], url: str)`` fits one whose class has an
        ``http_client``, of any type. On a registry that names no class, neither fits.
        """
        needs = context.cls.__name__
        if self.context_type is None:
            self.problems.append(problem.context_does_not_fit_no_context_type(node_id, context.name, needs=needs))
            return
        context_type = self.context_type.__name__
        if not is_protocol(context.cls):
            if not issubclass(self.context_type, context.cls):
                self.problems.append(problem.context_does_not_fit(node_id, context.name, needs=needs, context_type=context_type))
        elif lacks := what_the_context_lacks(self.context_type, context.cls):
            self.problems.append(problem.context_does_not_fit_lacks(node_id, context.name, needs=needs, context_type=context_type, lacks=lacks))

    def _compile_subgraph_node(self, node_id: str, record: NodeRecord, definition: SubgraphDefinition, edges: Mapping[str, tuple[Address, ...]]) -> None:
        """Compiles a node whose definition is a subgraph, and adds it to ``nodes``.

        Example: the subgraph ``strip -> shout`` used as the node ``loud``. The subgraph's graph record is
        compiled first, on its own. The node's inputs and outputs are the graph inputs and outputs its record
        declares, by name (``line``, ``result``), so an edge reads ``loud.result``. An output keeps the field the
        node inside declares. The loops and the branches an output depends on are
        asked of the compiled node (``loops_around``, ``branches_required_for``), which answers as it would if the
        subgraph's nodes were drawn in this graph.

        A subgraph that contains itself is a problem. So is every problem of the subgraph's own graph, with this
        node's id put in front of its ``loc``. In both cases the node is left out, because the inputs and outputs
        of a graph with problems are not known.
        """
        if definition.id in self.inside_subgraphs:
            self.problems.append(problem.subgraph_contains_itself(node_id, node=definition.id))
            return
        stack = (*self.inside_subgraphs, definition.id)
        key = (definition.id, definition.version, stack)
        if key not in self.compiled_subgraphs:
            self.compiled_subgraphs[key] = _Compiler(self.definitions, self.context_type, stack, self.compiled_subgraphs).compile(definition.graph, CompiledGraph)
        inner = self.compiled_subgraphs[key]
        if inner.problems:
            self.problems += [inner_problem.model_copy(update={"loc": (node_id, *inner_problem.loc)}) for inner_problem in inner.problems]
            return
        fields = inner._inputs
        planned = self._plan_inputs(node_id, record, fields, self._constants_of(node_id, record, fields), frozenset(), edges)
        self.nodes[node_id] = CompiledSubgraphNode(
            id=node_id, loops=planned.loops, inputs=planned.inputs, outputs={output: inner._output_field(output) for output in inner._outputs}, required_branches=planned.required_branches,
            definition=definition, inner=inner,
        )

    def _plan_inputs(
        self, node_id: str, record: NodeRecord, fields: Mapping[str, Field], constants: Mapping[str, Any], constant_at_compile: frozenset[str],
        edges: Mapping[str, tuple[Address, ...]], *, generic: bool = False,
    ) -> _InputPlans:
        """Makes the compiled input of every input of a node, and works out which loops the node is inside.

        In the example, ``shout.line`` reads one item of ``lines.result``, so ``shout`` is inside the loop over
        ``lines.result``. A node is inside the loops of its input with the most loops, and the loops of every
        other input must be the outermost of those (else ``series_not_nested``).

        An input that cannot get a value gets ``MissingInput``, and the graph has a problem that says why: an
        edge the input cannot read, loops that do not nest, or several edges that are not alternatives. The node
        is kept so an editor can still draw it; the graph does not run.

        Args:
            node_id: The node's id.
            record: The node in the graph record.
            fields: The node's inputs, by name.
            constants: The constant or default the record gives each input, already read by the input's type, by
                name; ``_INVALID`` where the type rejected it.
            constant_at_compile: The inputs a hook reads when the graph is compiled. Their value must be known
                then, so they never become graph inputs.
            edges: For each input that edges lead to, by name, the outputs they come from (``_edges_into``).
            generic: Whether an input's type has a type variable, which the edges then bind.
        """
        for name in sorted(record.inputs.keys() - fields.keys()):
            self.problems.append(problem.unknown_input(node_id, name, inputs=list(fields)))
        sources = {name: self._sources(node_id, name, edges[name]) for name in fields if name in edges}
        bound = None
        refused_defaults: set[str] = set()
        if generic:  # the compiled node's fields have no type variables left
            bound = _bound(fields, sources)
            fields, refused = _bind(node_id, fields, bound)
            for name, problems in refused.items():
                if name not in record.inputs:  # else the default is never used
                    self.problems += problems
                    refused_defaults.add(name)
        inputs: dict[str, Input] = {}
        readable: dict[str, list[Edge]] = {}  # the inputs whose edges can be read, planned once the node's loops are known
        collecting: set[str] = set()
        for name, field in fields.items():
            if name in refused_defaults:
                inputs[name] = MissingInput(field)
                continue
            match record.inputs.get(name):
                case RecordEdges():
                    edge_sources = sources[name]
                    if edge_sources is None:
                        inputs[name] = MissingInput(field)
                        continue
                    collects, expected = _collects_and_expected_type(field, edge_sources)
                    if self._edges_fit(node_id, field, expected, edge_sources):
                        readable[name] = edge_sources
                        if collects:
                            collecting.add(name)
                    else:
                        inputs[name] = MissingInput(field)
                case RecordConstant():
                    inputs[name] = _constant_input(field, constants[name])
                case RecordGraphInput() if name in constant_at_compile:  # a hook reads it before any caller gives it
                    self.problems.append(problem.hook_needs_value(node_id, name))
                    inputs[name] = MissingInput(field)
                case RecordGraphInput(graph_input=graph_input):
                    inputs[name] = self._graph_input(node_id, field, graph_input, constants.get(name, field.default))
                case None if field.default is REQUIRED and name not in constant_at_compile:  # an input a hook reads gets hook_needs_value instead
                    self.problems.append(problem.input_has_no_value(node_id, name))
                    inputs[name] = MissingInput(field)
                case None:
                    inputs[name] = ConstantInput(field, field.default)
        for name in self._inputs_with_conflicting_edges(node_id, fields, readable, collecting):
            del readable[name]
            inputs[name] = MissingInput(fields[name])
        # Only an edge puts a node inside a loop. A collecting input needs the loops around the one it collects.
        deepest = {  # most inputs have one edge
            name: edge_sources[0].loops_of_value if len(edge_sources) == 1 else max((s.loops_of_value for s in edge_sources), key=len)
            for name, edge_sources in readable.items()
        }
        loops = max((loops[:-1] if name in collecting else loops for name, loops in deepest.items()), key=len, default=())
        inputs = {
            name: _edge_input(fields[name], readable[name], deepest[name], loops, collects=name in collecting) if name in readable else inputs[name]
            for name in fields
        }
        for name, node_input in inputs.items():
            if loops[: len(node_input.loops)] != node_input.loops:  # not the node's outermost loops: items of two different lists are never paired
                series = [loops[-1], node_input.loops[-1]]
                reading = next(n for n, p in inputs.items() if p.loops == loops)
                subgraphs = list(dict.fromkeys(port.node for port in series if isinstance(self.nodes[port.node], CompiledSubgraphNode)))
                if subgraphs:  # an edge of this graph cannot change the loops inside a subgraph
                    self.problems.append(problem.loops_not_nested_inside_subgraph(
                        node_id, series=[str(s) for s in series], inputs=[reading, name], subgraphs=subgraphs,
                    ))
                else:
                    self.problems.append(problem.series_not_nested(
                        node_id, series=[str(s) for s in series], inputs=[reading, name],
                    ))
                inputs[name] = MissingInput(node_input.field)
        return _InputPlans(inputs, loops, self._required_branches(inputs), bound)

    def _graph_input(self, node_id: str, field: Field, name: str, default: Any) -> GraphInput | MissingInput:
        """Returns the compiled input of the input ``field`` of ``node_id``, whose value the caller gives as the graph
        input ``name``, and adds the graph input to ``graph_inputs``.

        Example: ``strip.line`` and ``shout.line`` both name the graph input ``line``, so a caller's ``line="a"`` is
        the value of both. The two must take one type and have one default; else the graph has the problem
        ``graph_input_used_two_ways`` and the second gets no value.

        Args:
            node_id: The node.
            field: The input.
            name: The graph input's name.
            default: The value when the caller gives none: the record's, else the input's own; ``_INVALID`` when the
                input's type refused the record's.
        """
        if default is _INVALID:
            return MissingInput(field)
        caller_field = Field(name, field.hint, default)
        found = self.graph_inputs.setdefault(name, caller_field)
        if found is not caller_field and (found.hint != caller_field.hint or found.default_json != caller_field.default_json):
            self.problems.append(problem.graph_input_used_two_ways(node_id, field.name, graph_input=name))
            return MissingInput(field)
        return GraphInput(field, name)

    def _constants_of(self, node_id: str, record: NodeRecord, fields: Mapping[str, Field]) -> dict[str, Any]:
        """Returns the constant, or the graph input's default, the record gives each of ``fields``, read by the input's
        type, by name. An input the record gives neither is left out. Where the type rejects the value, the graph gets
        the problem ``invalid_value`` and the input gets ``_INVALID``."""
        constants: dict[str, Any] = {}
        for name, field in fields.items():
            match record.inputs.get(name):
                case RecordConstant(constant=value) | RecordGraphInput(default=value) if value is not MISSING:
                    try:
                        constants[name] = read_constant(node_id, field, value)
                    except RefusedError as refused:
                        self.problems += refused.problems
                        constants[name] = _INVALID
        return constants

    def _hook(
        self, node_id: str, definition: FunctionDefinition[..., Any], kind: HookName, hook: Callable[..., Mapping[Any, Any]],
        constants: Mapping[str, Any], arriving: Mapping[str, Any], added: tuple[str, ...],
    ) -> dict[str, Field] | None:
        """Calls one of a node's hooks and returns what it returned, as fields by name; ``None``, with a
        problem, when the hook cannot be called, fails, or returns something that is not fields.

        ``@node`` read from the hook's signature what each parameter gets: the value of the input of the same
        name (its constant or default), the type the edges into that input read (``Any`` when no edge leads to
        it), or, for ``**added: type``, those types for every input ``compute_added_inputs`` added.

        Args:
            node_id: The node's id.
            definition: The node's definition.
            kind: Which hook it is: ``compute_added_inputs`` or ``compute_outputs``.
            hook: The hook.
            constants: The constant or default the record gives each input, read by the input's type, by name.
            arriving: The type the edges into each input read, by input name.
            added: The inputs ``compute_added_inputs`` added.
        """
        kwargs: dict[str, Any] = {}
        for name, receives in definition.hook_parameters[kind].items():
            match receives:
                case "arriving_type":
                    kwargs[name] = arriving.get(name, Any)
                case "added_arriving_types":
                    kwargs.update({input_name: arriving.get(input_name, Any) for input_name in added})
                case "value":
                    value = constants.get(name, definition.inputs[name].default)  # @node made sure the parameter names a declared input
                    if value is REQUIRED:
                        self.problems.append(problem.hook_needs_value(node_id, name))
                        return None
                    if value is _INVALID:
                        return None  # the problem invalid_value says why
                    kwargs[name] = value
        try:
            returned = hook(**kwargs)
        except NodeError as error:  # the hook raised a NodeError: keep its message, code and ctx
            self.problems.append(problem.compute_hook_failed(node_id, reason=error.message, error=error.code, error_ctx=dict(error.ctx) or None))
            return None
        except Exception as error:  # noqa: BLE001  any other exception of the hook is a problem of this node
            _logger.error("the %s hook of node %r failed", kind, node_id, exc_info=error)
            self.problems.append(problem.compute_hook_failed(node_id, reason=str(error), error=type(error).__name__))
            return None
        try:
            return read_hook_fields(node_id, kind, returned)
        except RefusedError as refused:
            self.problems += refused.problems
            return None

    # -- edges ------------------------------------------------------------------

    def _sources(self, node_id: str, name: str, addresses: tuple[Address, ...]) -> list[Edge] | None:
        """Returns what the edges from ``addresses`` into one input read, by the order of their nodes' ids and then
        of the outputs of each node. A graph output is read as the input ``name`` of ``"outputs"``.

        Returns:
            The sources. ``None`` when an edge cannot be read: its node is not in the graph record, or the address
            names no output of its node (the graph gets the problem), or its node was not compiled (a problem of
            its own says why, or it is on a cycle and comes later).
        """
        sources: list[Edge] = []
        for address in addresses:
            if address.node not in self.record.nodes:
                self.problems.append(problem.edge_names_missing_node(node_id, name, missing=address.node, source=str(address), target=f"{node_id}.{name}"))
                return None
            source = self._source(address, node_id, name)
            if source is None:
                return None
            sources.append(source)
        if len(sources) > 1:
            sources.sort(key=self._edge_order)
        return sources

    def _edges_fit(self, node_id: str, field: Field, expected: Type, sources: list[Edge]) -> bool:
        """Whether every edge into the input ``field`` reads a value of the type ``expected``; for the first that does
        not, the graph gets the problem.

        ``expected`` is the input's type when it takes one value, the type of one item when it is a ``Series`` that
        collects, and the record type when it is a ``Table`` that collects records (``_collects_and_expected_type``).
        """
        for source in sources:
            if expected is not source.type and expected.fits(source.type) is None:  # the identity test first: most edges read exactly the input's type
                self.problems.append(_mismatch(node_id, field, source))
                return False
        return True

    def _source(self, address: Address, node_id: str, name: str) -> Edge | None:
        """Returns what an edge from ``address`` reads: an output, or one field of the record an output returns.

        Returns:
            The edge's source, which reads one item per call when the output is a ``Series``. ``None``, with a
            problem, when the address names no output of its node; ``None`` when its node was not compiled.
        """
        source = self.nodes.get(address.node)
        if source is None:  # not compiled, or on a cycle and later: a problem of its own says why
            return None
        outputs = [str(Address(address.node, output)) for output in source.outputs]
        if address.output not in source.outputs:
            self.problems.append(problem.unknown_output(node_id, name, source=str(address), outputs=outputs))
            return None
        field = source.outputs[address.output]
        if address.field is not None and (parts := _taken_apart(field.type)):  # only a record has fields to wire; a whole table or list is taken apart first
            self.problems.append(problem.unknown_output_make_a_series(
                node_id, name, source=str(address), outputs=outputs, record_field=address.field, whole=parts[0], fields=parts[1],
            ))
            return None
        read_type = field.type if address.field is None else _field_type(field.type, address.field)
        if read_type is None:
            fields = [str(Address(address.node, address.output, f)) for f in field.type.fields]
            self.problems.append(problem.unknown_output(node_id, name, source=str(address), outputs=fields, record_field=address.field))
            return None
        return Edge(address, source.loops_around(address.output), read_type, item=field.is_series)

    def _edge_order(self, edge: Edge) -> tuple[int, int, str]:
        """Returns where an edge falls among several into one input: by its node's index among the node ids, then
        by its output's index in that node."""
        return self.position[edge.source.node], list(self.nodes[edge.source.node].outputs).index(edge.source.output), str(edge.source)

    # -- branches ---------------------------------------------------------------

    def _branches_required_for(self, edge: Edge) -> frozenset[RequiredBranch]:
        """The branches that must be chosen for the output ``edge`` reads to have a value."""
        return self.nodes[edge.source.node].branches_required_for(edge.source.output)

    def _inputs_with_conflicting_edges(
        self, node_id: str, fields: Mapping[str, Field], readable: Mapping[str, list[Edge]], collecting: set[str],
    ) -> set[str]:
        """Finds the inputs whose edges cannot be read together, and adds a problem for each.

        Several edges into one input must be alternatives: each pair must need different outputs of one exclusive
        group, so a call gets a value from at most one of them (else ``several_edges_into_one_input``). They must
        also be inside the same loops, so the one that has the value is found by the same item indexes (else
        ``alternatives_from_different_loops``).

        Two collecting inputs that do not accept ``None`` items, and whose edges are all on opposite branches, are
        a problem too (``lists_from_different_loops``). An item one list has no value for is left out of both
        lists, so both would always be empty.

        Args:
            node_id: The node.
            fields: The node's inputs, by name.
            readable: What the edges into each input read, for the inputs whose edges can be read, by name.
            collecting: The inputs among them whose type takes a list of what their edges read.

        Returns:
            The names of the inputs with a problem.
        """
        refused: set[str] = set()
        for name, sources in readable.items():
            branches = [self._branches_required_for(s) for s in sources]
            loops_of_inputs = list(dict.fromkeys(s.loops_of_value for s in sources))
            if not all(_never_together(a, b) for i, a in enumerate(branches) for b in branches[i + 1:]):
                self.problems.append(problem.several_edges_into_one_input(node_id, name, sources=[str(s.source) for s in sources]))
                refused.add(name)
            elif len(loops_of_inputs) > 1:
                series = [str(loops[-1]) if loops else None for loops in loops_of_inputs[:2]]
                self.problems.append(problem.alternatives_from_different_loops(node_id, name, series=series))
                refused.add(name)
        aligned = [  # the collecting inputs whose edges come from inside a loop
            (name, [self._branches_required_for(s) for s in sources]) for name, sources in readable.items()
            if name in collecting and any(s.loops_of_value for s in sources) and not fields[name].accepts_none_items and name not in refused
        ]
        for i, (a, branches_a) in enumerate(aligned):
            for b, branches_b in aligned[i + 1:]:
                if all(_never_together(x, y) for x in branches_a for y in branches_b):
                    self.problems.append(problem.lists_from_different_loops(node_id, inputs=[a, b]))
                    refused.update((a, b))
        return refused

    def _required_branches(self, inputs: Mapping[str, Input]) -> frozenset[RequiredBranch]:
        """Returns the branches that must be chosen for a node with these ``inputs`` to be called: for each
        input, the branches that every edge into it needs.

        A collecting input needs only the branches chosen outside the loop it collects. A branch chosen per
        item inside the loop only leaves items out of the list; the node is still called.
        """
        required_branches: set[RequiredBranch] = set()
        for node_input in inputs.values():
            if not node_input.sources:
                continue
            branches = [self._branches_required_for(s) for s in node_input.sources]
            shared = branches[0].intersection(*branches[1:])
            if isinstance(node_input, CollectingInput):
                shared = frozenset(branch for branch in shared if branch.depth <= len(node_input.loops))
            required_branches |= shared
        return frozenset(required_branches)


def _edges_into(record: NodeRecord) -> dict[str, tuple[Address, ...]]:
    """The outputs the edges into each input of a node come from, by input name: in the example, ``shout``'s are
    ``{"line": (Address("lines", "result"),)}``. An input with a constant or a default is not in it."""
    edges: dict[str, tuple[Address, ...]] = {}
    for name, entry in record.inputs.items():
        match entry:
            case RecordEdges(sources=sources):
                edges[name] = sources
    return edges


def _collects_and_expected_type(field: Field, sources: list[Edge]) -> tuple[bool, Type]:
    """Returns whether the input ``field`` collects what ``sources`` read, and the type each source must fit:
    the input's whole type when it reads one value; when it collects, the type of one item of the ``Series`` or of
    one record of the table.

    A ``Table`` input reads a whole table when every edge reads one, as one value or as each item of a ``Series``
    of tables. An output declared ``Any`` could be a table or a record: it counts as a table when it has one value
    in the run, and as a record when it has one value per item of a loop.
    """
    record = field.whole.record
    collects = field.is_series or (record is not None and not all(
        field.whole.fits(s.type) is not None and (s.type.bare is not Any or not s.loops_of_value) for s in sources
    ))
    return collects, field.whole if not collects else field.type if record is None else record


def _edge_input(field: Field, sources: list[Edge], deepest: tuple[Address, ...], loops: tuple[Address, ...], *, collects: bool) -> EdgeInput | CollectingInput:
    """Returns the compiled input of an input whose edges read ``sources``, for a node inside ``loops``.

    In the example, ``shout.line`` reads one item of ``lines.result`` per call, and ``join.texts`` collects what
    ``shout`` returned per line, so ``join`` is outside the loop over ``lines.result``. Wired to an output of a
    node inside no loop, ``join.texts`` gets that one value as a list of one. So does a collecting input whose node
    another input already puts inside the loop it would collect: each call then has one value of that loop, its
    own item's.

    Args:
        field: The input.
        sources: What its edges read; all are inside the same loops.
        deepest: The loops a node must be inside to read one value of a source: those of the source with the most.
        loops: The loops the node is inside.
        collects: Whether the input's type takes a list of what the edges read (a ``Series``, or a ``Table`` of
            records), rather than one value of it.
    """
    if not collects:
        return EdgeInput(field, tuple(sources), deepest)
    if deepest == loops[: len(deepest)] and not any(s.item for s in sources):
        return EdgeInput(field, tuple(sources), deepest, as_list=True)
    return CollectingInput(field, deepest[-1], deepest[:-1], tuple(sources))


def _bound(fields: Mapping[str, Field], sources: Mapping[str, list[Edge] | None]) -> dict[TypeVar, Any]:
    """Returns the type each type variable of a node's inputs gets from the edges into them. When edges into
    two inputs that share a variable read ``int`` and ``str``, the variable gets ``int | str``.

    Only an edge binds a variable: a constant, a default and a graph input bind nothing. A variable no edge binds
    becomes its bound, else ``Any``, when the fields are bound.
    """
    bound: dict[TypeVar, Any] = {}
    for name, field in fields.items():
        edge_sources = sources.get(name)
        if not field.whole.variables or not edge_sources:
            continue
        expected = _collects_and_expected_type(field, edge_sources)[1]
        for source in edge_sources:
            for variable, hint in (expected.fits(source.type) or {}).items():
                bound[variable] = bound[variable] | hint if variable in bound else hint
    return bound


def _bind(node_id: str, fields: Mapping[str, Field], bound: Mapping[TypeVar, Any]) -> tuple[dict[str, Field], dict[str, list[Problem]]]:
    """Returns the inputs or outputs ``fields`` of the node ``node_id`` with their type variables bound as ``bound``
    says (``bound_field``), and the problem of each whose declared default the bound type refuses, by name. Such a
    field keeps no default."""
    bound_fields: dict[str, Field] = {}
    refused: dict[str, list[Problem]] = {}
    for name, field in fields.items():
        try:
            bound_fields[name] = bound_field(node_id, field, bound)
        except RefusedError as error:
            bound_fields[name] = Field(name, Type(field.hint).bind(bound).hint)
            refused[name] = error.problems
    return bound_fields, refused


def _taken_apart(whole: Type) -> tuple[Literal["table", "list"], list[str]] | None:
    """Returns what a value of the type ``whole`` gives when a node takes it apart into a ``Series`` of its rows or
    items, for a problem to suggest that node.

    Returns:
        ``"table"`` or ``"list"``, and the fields of each record or item it gives: none for a table whose columns
        are not known, and none for a list whose items are not records. ``None`` for a type that is neither a
        table nor a list.
    """
    if whole.record is not None:
        return "table", list(whole.fields)
    one = whole.options[0] if len(whole.options) == 1 else None
    if one is list or typing.get_origin(one) is list:
        return "list", list(Type(next(iter(typing.get_args(one)), Any)).fields)
    return None


def _mismatch(node_id: str, field: Field, source: Edge) -> Problem:
    """Returns the ``type_mismatch`` problem of an edge from ``source`` into the input ``field``.

    Where one mistake explains it, the problem says the fix: a whole table or list wired to an input that takes
    one value needs a node first that takes it apart into a ``Series``, and a table input wired to records that lack one of its columns needs
    that column added.
    """
    address, expected = str(source.source), field.whole.id
    got = f"Series[{source.type.text}]" if source.item else source.type.text
    parts = _taken_apart(source.type)
    if field.whole.record is not None:
        has = (source.type.record or source.type).fields
        missing = [column for column in field.whole.fields if column not in has]
        if has and missing:
            column_type = Type(field.whole.fields[missing[0]]).id
            return problem.type_mismatch_add_column(node_id, field.name, source=address, got=got, expected=expected, column=missing[0], column_type=column_type, has=list(has))
    elif parts and not field.is_series:
        return problem.type_mismatch_make_a_series(node_id, field.name, source=address, got=got, expected=expected, whole=parts[0], fields=parts[1])
    return problem.type_mismatch(node_id, field.name, source=address, got=got, expected=expected)


def _constant_input(field: Field, value: Any) -> ConstantInput | MissingInput:
    """Returns the compiled input of an input with a constant: ``ConstantInput``, or ``MissingInput`` when the input's type
    rejected the constant."""
    return MissingInput(field) if value is _INVALID else ConstantInput(field, value)


def _field_type(record: Type, name: str) -> Type | None:
    """Returns the type of the field ``name`` of a record of the type ``record``.

    Returns:
        A ``TypedDict``'s, dataclass's or pydantic model's declared type for it; a mapping's value type, ``Any``
        for a bare ``dict`` and for a value of any type; ``None`` for a record without the field, and for a type
        that is not a record.
    """
    one = record.options[0] if len(record.options) == 1 else None
    if one is Any or one is dict:
        return Type(Any)
    if typing.get_origin(one) in (dict, Mapping):
        return Type(typing.get_args(one)[1])
    return Type(record.fields[name]) if name in record.fields else None


def _never_together(a: frozenset[RequiredBranch], b: frozenset[RequiredBranch]) -> bool:
    """Whether two sets of required branches can never both be chosen: one exclusive group must choose one output
    for ``a`` and another for ``b``."""
    return any(x[:2] == y[:2] and x.output != y.output for x in a for y in b)


def _reaches(start: str, goal: str, downstream: Mapping[str, list[str]], within: set[str]) -> bool:
    """Whether ``goal`` is downstream of ``start``, following only edges through the nodes in ``within``."""
    seen: set[str] = set()
    stack = list(downstream[start])
    while stack:
        n = stack.pop()
        if n == goal:
            return True
        if n in within and n not in seen:
            seen.add(n)
            stack.extend(downstream[n])
    return False
