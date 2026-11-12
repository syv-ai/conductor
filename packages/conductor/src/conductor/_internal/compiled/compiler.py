"""Compiling: a graph record checked against the nodes it uses, into a ``CompiledGraph``.

Loading does everything once: every edge and value is checked, and each node
gets the scope it runs in. What is wrong becomes a ``Problem`` with a place,
never an exception. Declared types alone decide every scope, by three rules:

- an input of one ``T`` fed by a ``Series[T]`` reads one item, inside the map
  that Series opens; a ``Table`` input fed a ``Series`` of tables is one;
- an input fed from an enclosing scope captures that scope's one value;
- an input of ``Series[T]``, or of ``Table[R]`` fed records, gathers: it
  receives every value its source made: the items of the innermost map of
  what feeds it, in order, and it sits in the scope that map's Series was
  made in; or, from a source that runs once, that one value as a Series of
  one.

A node runs in the deepest of its inputs' scopes, and every input's scope
lies on the path to it: items of two Series never pair up
(``series_not_nested``). Readers of one Series read one map.
"""

from __future__ import annotations

import heapq
import logging
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypeVar

from pydantic import StringConstraints, TypeAdapter, ValidationError

from conductor._internal.address import split_address
from conductor._internal.compiled.describe import GraphInput
from conductor._internal.compiled.input_plans import (
    EdgeSource,
    InputCollectsLoopResults,
    InputConstant,
    InputFromEdges,
    InputFromGraphInput,
    InputMissing,
    InputPlan,
    InputReadsOneValue,
    InputWrapsOneValue,
    OutputAddress,
    OutputInLoops,
)
from conductor._internal.compiled.nodes import (
    CompiledFunctionNode,
    CompiledNode,
    CompiledSubgraphNode,
    RequiredBranch,
)
from conductor._internal.compiled.type import CompiledGraph, InputField
from conductor._internal.field import Field
from conductor._internal.graph_record import GraphRecord, NodeRecord
from conductor._internal.markers import REQUIRED
from conductor._internal.node_definition import (
    FunctionDefinition,
    HookName,
    NodeDefinition,
    SubgraphDefinition,
    is_node_id,
    read_by_hooks,
)
from conductor._internal.problem import NodeDeclarationError, NodeError, Problem
from conductor._internal.problem import catalogue as problem
from conductor._internal.value_type import FieldValueError, Param, Type

_logger = logging.getLogger("conductor")


def compile_graph[GraphT: CompiledGraph](record: GraphRecord, definitions: Mapping[str, NodeDefinition], graph_type: type[GraphT]) -> GraphT:
    """``record`` loaded against ``definitions``, the newest version of each node by id, as a ``graph_type``: the
    ``Graph`` a host holds, which is the one way one is made."""
    return _Compiler(definitions, (), {}).compile(record, graph_type)



#: What stands for a typed-in value its field's type refused, while a node is placed.
_INVALID: Any = object()

#: Reads what a hook returns: fields by name, a name holding none of the marks an address is written with.
_HOOK_FIELDS = TypeAdapter(dict[Annotated[str, StringConstraints(min_length=1, pattern=r"^[^./@]+$")], Any])


@dataclass(frozen=True, slots=True)
class _InputPlans:
    """A node's inputs, planned.

    Attributes:
        inputs: How each input gets its value, by name.
        loops: The loops the node is inside, outermost first.
        required_branches: The branches that must be chosen upstream for the node to be called.
        bound: What the node's type variables are bound to by what arrives on its edges; ``None`` for a node whose
            inputs hold no type variable, which is nearly every node, so nothing is bound or replaced for it.
    """

    inputs: dict[str, InputPlan]
    loops: tuple[OutputAddress, ...]
    required_branches: frozenset[RequiredBranch]
    bound: Mapping[TypeVar, Any] | None


class _Compiler:
    """Compiles one graph record. It holds what is known so far while the nodes are compiled one at a time,
    each after the nodes upstream of it.

    Attributes:
        definitions: The newest version of each node definition, by id.
        inside_subgraphs: The ids of the subgraph definitions this record is inside, outermost first, so a
            subgraph that contains itself is found.
        alone: Each subgraph compiled so far. Every compiler made for one ``compile_graph`` call shares it, so
            a subgraph used twice is compiled once.
        problems: The problems found so far.
        nodes: Each node compiled so far, by id.
        opened: Each graph input found so far, with its field, in the order found.
        position: Each node's index in the record.
    """

    def __init__(self, definitions: Mapping[str, NodeDefinition], inside_subgraphs: tuple[str, ...], alone: dict[tuple[str, int, tuple[str, ...]], CompiledGraph]) -> None:
        self.definitions, self.inside_subgraphs, self.alone = definitions, inside_subgraphs, alone
        self.problems: list[Problem] = []
        self.nodes: dict[str, CompiledNode] = {}
        self.opened: list[tuple[GraphInput, Field]] = []
        self.position: dict[str, int] = {}

    def compile[GraphT: CompiledGraph](self, record: GraphRecord, graph_type: type[GraphT]) -> GraphT:
        """Returns ``record`` compiled, as a ``graph_type``. It never raises for a wrong record: the graph has
        the problems, and every node whose inputs and outputs could be worked out."""
        refused = {n.id for n in record.nodes if not is_node_id(n.id)}
        self.problems += [problem.invalid_node_id(node_id) for node_id in dict.fromkeys(n.id for n in record.nodes) if node_id in refused]
        known: dict[str, NodeRecord] = {}
        twice: set[str] = set()
        for node_record in record.nodes:
            if node_record.id in known:
                twice.add(node_record.id)
            elif node_record.id not in refused:
                known[node_record.id] = node_record
        self.problems += [problem.duplicate_node_id(node_id) for node_id in sorted(twice)]
        self.position = {node_id: i for i, node_id in enumerate(known)}
        into: dict[str, dict[str, list[str]]] = {node_id: {} for node_id in known}
        seen: set[tuple[str, str]] = set()
        for source, target in record.edges:
            node_id, name = split_address(target)
            source_id = split_address(source)[0]
            if source_id in refused or node_id in refused:
                continue  # its own invalid_node_id says why
            if (source, target) in seen:
                self.problems.append(problem.repeated_edge(node_id, name, source=source))
                continue
            seen.add((source, target))
            missing = [n for n in (source_id, node_id) if n not in known]
            if missing:
                self.problems.append(problem.edge_names_missing_node(node_id, name, missing=missing[0], source=source, target=target))
            if node_id in known:  # the input is drawn as fed, whether or not its source exists
                into[node_id].setdefault(name, []).append(source)
        ordered = self._order(list(known), into)
        for node_id in ordered:
            match self._resolve(known[node_id]):
                case FunctionDefinition() as function:
                    self._compile_function_node(known[node_id], function, into[node_id])
                case SubgraphDefinition() as graph:
                    self._compile_subgraph_node(known[node_id], graph, into[node_id])
                case None:
                    pass
        names = [field.name for _, field in self.opened]
        inputs: dict[str, GraphInput] = {}
        input_fields: dict[str, InputField] = {}
        for graph_input, field in self.opened:
            name = field.name if names.count(field.name) == 1 else graph_input.address
            inputs[name] = graph_input
            input_fields[graph_input.address] = InputField(name, graph_input.address, field)
        downstream: dict[str, list[CompiledNode]] = {node_id: [] for node_id in self.nodes}
        needs_context: list[tuple[str, FunctionDefinition[..., Any]]] = []
        for compiled_node in self.nodes.values():
            needs_context += compiled_node.functions_with_context
            for source_id in compiled_node.upstream_ids:
                downstream[source_id].append(compiled_node)
        return graph_type(record, self.problems, inputs, self.nodes, input_fields, downstream, needs_context)

    def _resolve(self, record: NodeRecord) -> NodeDefinition | None:
        """The definition a node of the record names, at its version; ``None`` with the problem when there is none."""
        newest = self.definitions.get(record.node)
        if newest is None:
            self.problems.append(problem.unknown_node(record.id, node=record.node, choices=sorted(self.definitions)))
            return None
        if record.version not in newest.versions:
            self.problems.append(problem.unknown_version(record.id, node=record.node, version=record.version, newest=newest.version))
            return None
        return newest.version_of(record.version)

    def _order(self, order: list[str], into: dict[str, dict[str, list[str]]]) -> list[str]:
        """The nodes with each after what its edges read, ties in the record's order.

        A node on a cycle is a problem. The first of a cycle's nodes in the record is placed before the node
        whose edge closes the cycle, so that edge brings it nothing, and the rest follow in order.
        """
        position = self.position
        reads = {n: frozenset({split_address(s)[0] for sources in into[n].values() for s in sources} & position.keys()) for n in order}
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
            if not ready:  # every node left waits on a cycle
                if on_cycles is None:
                    stuck = {n for n in order if waiting[n]}
                    on_cycles = [n for n in order if n in stuck and _reaches(n, n, downstream, stuck)]  # on a cycle, not only below one
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

    # -- placing a node ---------------------------------------------------------

    def _compile_function_node(self, record: NodeRecord, definition: FunctionDefinition[..., Any], edges: dict[str, list[str]]) -> None:
        """Compiles a node that is a function and adds it to ``nodes``.

        Its inputs are the declared ones and those its ``compute_added_inputs`` hook adds; each gets an input
        plan. Its outputs are the declared ones, or what its ``compute_outputs`` hook returns for the types its
        edges read. A node whose hook fails is left out, with the problem. Any other node is added, whatever
        problems its inputs have.
        """
        fields = dict(definition.inputs)
        constants = self._constants_of(record, fields)
        added: dict[str, Field] = {}
        if definition.compute_added_inputs is not None:
            made = self._hook(record, definition, "compute_added_inputs", definition.compute_added_inputs, constants, {}, ())
            if made is None:
                return
            added = made
            taken = sorted(added.keys() & (fields.keys() | definition.context.keys()))
            if taken:
                self.problems.append(problem.compute_hook_failed(record.id, reason=f"it adds {taken[0]!r}, which the node already declares", error="added_input_taken"))
                return
            fields.update(added)
            constants.update(self._constants_of(record, added))
        planned = self._plan_inputs(record, fields, constants, read_by_hooks(definition), edges, generic=definition.generic)
        bound = planned.bound
        outputs: dict[str, Field] | None = dict(definition.outputs) if bound is None else {name: field.bind(bound) for name, field in definition.outputs.items()}
        if definition.compute_outputs is not None:
            arriving = {name: plan.sources[0].type.bare for name, plan in planned.inputs.items() if plan.sources}
            outputs = self._hook(record, definition, "compute_outputs", definition.compute_outputs, constants, arriving, tuple(added))
        if outputs is not None:
            self.nodes[record.id] = CompiledFunctionNode(
                id=record.id, loops=planned.loops, inputs=planned.inputs, outputs=outputs, required_branches=planned.required_branches,
                definition=definition,
            )

    def _compile_subgraph_node(self, record: NodeRecord, definition: SubgraphDefinition, edges: dict[str, list[str]]) -> None:
        """Compiles a node that is a subgraph and adds it to ``nodes``.

        Example: the subgraph ``strip -> shout`` used as the node ``loud``. The subgraph's record is compiled on
        its own first. The node's inputs are that graph's inputs under their names there (``line``), and its
        outputs are that graph's outputs under their addresses there, so an edge reads ``loud.shout.result``.
        An output keeps the field the node inside declares. What an edge outside needs to know about it is
        asked of the node: ``loops_around`` for the loops inside it is returned in, ``branches_required_for``
        for the branches inside it depends on. Both answer as the same nodes drawn in this graph would.

        A subgraph that contains itself is a problem. The problems of the subgraph are problems of this graph,
        each with this node's id in front of its ``loc``. Either way the node is left out, as the inputs and
        outputs of a graph with problems are not known.
        """
        if definition.id in self.inside_subgraphs:
            self.problems.append(problem.subgraph_contains_itself(record.id, node=definition.id))
            return
        stack = (*self.inside_subgraphs, definition.id)
        key = (definition.id, definition.version, stack)
        if key not in self.alone:
            self.alone[key] = _Compiler(self.definitions, stack, self.alone).compile(definition.graph, CompiledGraph)
        inner = self.alone[key]
        if inner.problems:
            self.problems += [problem.model_copy(update={"loc": (record.id, *problem.loc)}) for problem in inner.problems]
            return
        fields = {graph_input.name: Field(graph_input.name, graph_input.field.hint, graph_input.field.default) for graph_input in inner._input_fields.values()}
        outputs: dict[str, Field] = {}
        for address in inner.outputs:
            node_id, output = split_address(address)
            at = inner._nodes[node_id]
            field = at.outputs[output]
            group = field.param.exclusive_group
            if group:  # a decision inside is this node's, named by where it is made
                field = Field(field.name, Annotated[field.hint, Param(exclusive_group=f"{node_id}.{group}")], field.default)
            outputs[address] = Field(address, field.hint, field.default)
        planned = self._plan_inputs(record, fields, self._constants_of(record, fields), frozenset(), edges)
        self.nodes[record.id] = CompiledSubgraphNode(
            id=record.id, loops=planned.loops, inputs=planned.inputs, outputs=outputs, required_branches=planned.required_branches,
            definition=definition, inner=inner,
        )

    def _plan_inputs(
        self, record: NodeRecord, fields: Mapping[str, Field], constants: Mapping[str, Any], constant_at_compile: frozenset[str],
        edges: dict[str, list[str]], *, generic: bool = False,
    ) -> _InputPlans:
        """Makes the input plan of every input of a node, and works out which loops the node is inside.

        In ``lines -> shout -> join``: ``shout.line`` reads one item of ``lines.result``, so ``shout`` is inside
        the loop over ``lines.result``. A node is inside the loops of the input with the most loops, and the
        loops of every other input must be the outermost of those (else ``series_not_nested``).

        An input that cannot get a value (an edge it cannot read, loops that do not nest, edges that are not
        alternatives) gets ``InputMissing``, with the problem. The node is kept so an editor can draw it; the
        graph never runs.

        Args:
            record: The node in the record.
            fields: The node's inputs, by name.
            constants: The constant and the default the record has for each input, already read by the input's
                type, by name.
            constant_at_compile: The inputs a hook reads when the graph is compiled. They never become graph
                inputs.
            edges: The addresses of the outputs wired into each input, by input name.
            generic: Whether an input's type has a type variable, which the edges then bind.
        """
        for name in sorted({*record.constants, *record.defaults, *edges} - fields.keys()):
            self.problems.append(problem.unknown_input(record.id, name, inputs=list(fields)))
        sources = {name: self._sources(record.id, name, edges[name]) for name in fields if name in edges}
        bound = None
        if generic:
            bound = _bound(fields, sources)
            fields = {name: field.bind(bound) for name, field in fields.items()}  # a placed node's fields hold no type variable
        first = not any(name in fields for name in edges)  # a first node: nothing is wired into it, so the caller fills it
        inputs: dict[str, InputPlan] = {}
        for name, field in fields.items():
            if name in record.constants and name in record.defaults:
                self.problems.append(problem.constant_and_default(record.id, name))
            if name in edges:
                if name in record.constants or name in record.defaults:
                    self.problems.append(problem.constant_and_edge(record.id, name))
                source_kind = sources[name]
                inputs[name] = InputMissing(field) if source_kind is None else self._plan_from_edges(record.id, field, source_kind)
            elif name in record.constants:
                inputs[name] = _constant_plan(field, constants[name])
            elif first and name not in constant_at_compile:
                default = constants.get(name, field.default)
                if default is _INVALID:
                    inputs[name] = InputMissing(field)
                    continue
                address = f"{record.id}.{name}"
                opened = field if default is field.default else Field(name, field.hint, default)  # the record's default is the caller's
                self.opened.append((GraphInput(address=address, node_id=record.id, field=opened.description), opened))
                inputs[name] = InputFromGraphInput(field, address)
            elif name in record.defaults:
                self.problems.append(problem.default_on_later_node(record.id, name))
                inputs[name] = _constant_plan(field, constants[name])
            elif field.default is REQUIRED and name not in constant_at_compile:  # a hook that reads it says so itself, as hook_needs_value
                self.problems.append(problem.unbound_on_later_node(record.id, name))
                inputs[name] = InputMissing(field)
            else:
                inputs[name] = InputConstant(field, field.default)
        for name in self._never_meet(record.id, inputs):
            inputs[name] = InputMissing(fields[name])
        loops = max((plan.loops for plan in inputs.values()), key=len, default=())
        for name, plan in inputs.items():
            if isinstance(plan, InputCollectsLoopResults) and (made_in := plan.sources[0].loops_of_value) == loops[: len(made_in)] and not any(s.item for s in plan.sources):
                # its source makes one value where the node runs, for another input's sake: that one, never every item's
                inputs[name] = InputWrapsOneValue(plan.field, plan.sources)
        for name, plan in inputs.items():
            if loops[: len(plan.loops)] != plan.loops:  # the input's loops are not the node's outermost loops: items of two lists never pair up
                series = [loops[-1], plan.loops[-1]]
                outer, inner = sorted(series, key=lambda port: list(self.nodes).index(port.node_id))  # the one placed later can read the other with no cycle
                reading = next(n for n, p in inputs.items() if p.loops == loops)
                subgraphs = list(dict.fromkeys(port.node_id for port in series if isinstance(self.nodes[port.node_id], CompiledSubgraphNode)))
                if subgraphs:  # an edge of this graph can't move a list inside a subgraph
                    self.problems.append(problem.loops_not_nested_inside_subgraph(
                        record.id, series=[str(s) for s in series], inputs=[reading, name], subgraphs=subgraphs,
                    ))
                else:
                    self.problems.append(problem.series_not_nested(
                        record.id, series=[str(s) for s in series], inputs=[reading, name],
                        wire_from=str(outer), wire_into=inner.node_id,
                    ))
                inputs[name] = InputMissing(plan.field)
        return _InputPlans(inputs, loops, self._required_branches(inputs), bound)

    def _constants_of(self, record: NodeRecord, fields: Mapping[str, Field]) -> dict[str, Any]:
        """Each value typed into the record for one of ``fields``, and else each default, as the field reads it;
        ``_INVALID``, with the problem, where it can't."""
        constants: dict[str, Any] = {}
        for name, field in fields.items():
            if name not in record.constants and name not in record.defaults:
                continue
            try:
                constants[name] = field.whole.read(record.constants[name] if name in record.constants else record.defaults[name])
            except FieldValueError as refused:
                self.problems.append(problem.invalid_value(record.id, name, reason=refused.reason, errors=refused.errors, column=refused.column))
                constants[name] = _INVALID
        return constants

    def _hook(
        self, record: NodeRecord, definition: FunctionDefinition[..., Any], kind: HookName, hook: Callable[..., Mapping[Any, Any]],
        constants: Mapping[str, Any], arriving: Mapping[str, Any], added: tuple[str, ...],
    ) -> dict[str, Field] | None:
        """Call a hook with what its parameters read: its answer read as fields, or ``None`` with the problem.

        Args:
            record: The node in the record.
            definition: The node's definition.
            kind: Which hook it is.
            hook: The hook.
            constants: Each value and default typed into the record, as its field read it, by input.
            arriving: The type arriving on each input edges feed, by input; ``Any`` arrives on any other.
            added: The inputs ``compute_added_inputs`` added, which ``**added: type`` reads the arriving types of.
        """
        kwargs: dict[str, Any] = {}
        for name, receives in definition.hook_parameters[kind].items():
            match receives:
                case "arriving_type":
                    kwargs[name] = arriving.get(name, Any)
                case "added_arriving_types":
                    kwargs.update({input_name: arriving.get(input_name, Any) for input_name in added})
                case "value":
                    value = constants.get(name, definition.inputs[name].default)  # @node checked the parameter names a declared input
                    if value is REQUIRED:
                        self.problems.append(problem.hook_needs_value(record.id, name))
                        return None
                    if value is _INVALID:
                        return None  # its own invalid_value says why
                    kwargs[name] = value
        try:
            returned = hook(**kwargs)
        except NodeError as error:  # the hook's own account, with its own code and its ctx kept whole
            self.problems.append(problem.compute_hook_failed(record.id, reason=error.message, error=error.code, error_ctx=dict(error.ctx) or None))
            return None
        except Exception as error:  # noqa: BLE001  the hook's failure is this node's problem
            _logger.error("the %s hook of node %r failed", kind, record.id, exc_info=error)
            self.problems.append(problem.compute_hook_failed(record.id, reason=str(error), error=type(error).__name__))
            return None
        try:
            hints = _HOOK_FIELDS.validate_python(returned)
        except ValidationError as invalid:
            for fault in invalid.errors():
                if fault["loc"]:  # a name refused: the place is the name
                    self.problems.append(problem.parameter_name_invalid(record.id, str(fault["loc"][0])))
                else:  # the value itself refused: it is no mapping
                    reason = f"its {kind} hook returned a {type(returned).__name__}; a hook returns its fields by name, as a mapping of each name to its type"
                    self.problems.append(problem.compute_hook_failed(record.id, reason=reason, error="not_a_mapping"))
            return None
        try:
            return {name: Field(name, *_hint_and_default(hint)) for name, hint in hints.items()}
        except NodeDeclarationError as error:  # a hint conductor can't carry is the hook's failure
            self.problems.append(problem.compute_hook_failed(record.id, reason=error.message, error=error.code))
            return None

    # -- edges ------------------------------------------------------------------

    def _sources(self, node_id: str, name: str, addresses: list[str]) -> list[EdgeSource] | None:
        """What the edges from ``addresses`` bring one input, in the order their nodes and outputs have in the record.

        Returns:
            ``None`` when an edge names nothing a node gives, with the problem, and ``None`` when its node's
            fields aren't known or it is placed later, on a cycle.
        """
        sources: list[EdgeSource] = []
        for address in addresses:
            source = self._source(address, node_id, name)
            if source is None:
                return None
            sources.append(source)
        if len(sources) > 1:
            sources.sort(key=self._edge_order)
        return sources

    def _plan_from_edges(self, node_id: str, field: Field, sources: list[EdgeSource]) -> InputFromEdges | InputMissing:
        """Returns the plan of an input whose edges read ``sources``.

        An input of one value reads one value per call, whatever its type; when a source is a ``Series`` that
        is one item of it. A ``Series`` input collects the innermost loop its sources are made in, or wraps the
        one value of a source made in no loop. A ``Table`` input reads a table when every edge reads one (a
        ``Series`` of tables one table at a time), and else collects records.

        Returns:
            The plan. ``InputMissing``, with the problem, when an edge reads a type the input does not accept.
        """
        collects, expected = _collects_and_expected_type(field, sources)
        for source in sources:
            if expected is not source.type and expected.fits(source.type) is None:  # the same type, which most edges join, fits
                self.problems.append(_mismatch(node_id, field, source))
                return InputMissing(field)
        deepest = max((s.loops_of_value for s in sources), key=len)
        if not collects:
            return InputReadsOneValue(field, tuple(sources), deepest)
        if not deepest:
            return InputWrapsOneValue(field, tuple(sources))
        source_node, output = deepest[-1]
        return InputCollectsLoopResults(field, OutputInLoops(source_node, output, deepest[:-1]), tuple(sources))

    def _source(self, address: str, node_id: str, name: str) -> EdgeSource | None:
        """What an edge from ``address`` brings: ``node.output``, or ``node.output.field`` for one field of a record.

        The output is the one of the node's whose name the address goes on with after the node's id: a
        subgraph's is itself an address, ``e.upper.result``.

        Returns:
            An item of a ``Series`` output, or an output's one value; ``None`` with the problem when the address
            names nothing its node gives, and ``None`` when its node isn't placed.
        """
        source_id, rest = split_address(address)
        source = self.nodes.get(source_id)
        if source is None:  # a node whose fields aren't known, one placed later on a cycle, or none: its own problem says why
            return None
        output, _, expected = rest.partition(".")
        if output not in source.outputs:  # a subgraph's output: the longest part of the address that names one
            output = next((declared for declared in source.outputs if rest == declared or rest.startswith(f"{declared}.")), "")
            expected = rest[len(output) + 1 :]
        field = source.outputs.get(output)
        if field is None:
            self.problems.append(problem.unknown_output(node_id, name, source=address, outputs=[f"{source_id}.{o}" for o in source.outputs]))
            return None
        if expected and (split := _split_gives(field.type)):  # a table or a list is one value: only an item of a Series has a field to wire
            self.problems.append(problem.unknown_output_split_first(
                node_id, name, source=address, outputs=[f"{source_id}.{o}" for o in source.outputs], record_field=expected,
                whole=split[0], fields=split[1],
            ))
            return None
        sources = _field_type(field.type, expected) if expected else field.type
        if sources is None:
            fields = [f"{source_id}.{output}.{f}" for f in field.type.fields]
            self.problems.append(problem.unknown_output(node_id, name, source=address, outputs=fields, record_field=expected or None))
            return None
        return EdgeSource(source_id, output, source.loops_around(output), expected or None, sources, item=field.is_series)

    def _edge_order(self, source: EdgeSource) -> tuple[int, int, str]:
        """Where an edge into an input falls among several: its node's place in the record, then the output's in its node."""
        return self.position[source.node_id], list(self.nodes[source.node_id].outputs).index(source.output), source.address

    # -- branches ---------------------------------------------------------------

    def _branches_required_for(self, source: EdgeSource) -> frozenset[RequiredBranch]:
        """The branches that must be chosen for the output ``source`` reads to have a value."""
        return self.nodes[source.node_id].branches_required_for(source.output)

    def _never_meet(self, node_id: str, inputs: Mapping[str, InputPlan]) -> set[str]:
        """Finds the inputs whose edges cannot be read together, and adds a problem for each.

        Several edges into one input must be alternatives. Each pair must need different outputs of one
        exclusive group, so that a call gets a value from at most one of them
        (``several_edges_into_one_input``). And all must be made in the same loops, so that whichever has the
        value is found by the same item indexes (``alternatives_from_different_loops``).

        Two collecting inputs that accept no ``None`` items and whose edges are all on opposite branches would
        each drop the items the other keeps, leaving both lists empty (``lists_from_different_loops``).

        Returns:
            The names of the inputs with a problem.
        """
        refused: set[str] = set()
        for name, plan in inputs.items():
            behind = [self._branches_required_for(s) for s in plan.sources]
            loops_of_inputs = list(dict.fromkeys(s.loops_of_value for s in plan.sources))
            if not all(_opposed(a, b) for i, a in enumerate(behind) for b in behind[i + 1:]):
                self.problems.append(problem.several_edges_into_one_input(node_id, name, sources=[s.address for s in plan.sources]))
                refused.add(name)
            elif len(loops_of_inputs) > 1:
                series = [str(loops[-1]) if loops else None for loops in loops_of_inputs[:2]]
                self.problems.append(problem.alternatives_from_different_loops(node_id, name, series=series))
                refused.add(name)
        aligned = [
            (name, [self._branches_required_for(s) for s in plan.sources]) for name, plan in inputs.items()
            if isinstance(plan, InputCollectsLoopResults) and not plan.field.accepts_none_items and name not in refused
        ]
        for i, (a, behind_a) in enumerate(aligned):
            for b, behind_b in aligned[i + 1:]:
                if all(_opposed(x, y) for x in behind_a for y in behind_b):
                    self.problems.append(problem.lists_from_different_loops(node_id, inputs=[a, b]))
                    refused.update((a, b))
        return refused

    def _required_branches(self, inputs: Mapping[str, InputPlan]) -> frozenset[RequiredBranch]:
        """Returns the branches that must be chosen for a node with these ``inputs`` to be called: for each
        input, the branches that every edge into it needs.

        A collecting input needs only the branches chosen outside the loop it collects. A branch chosen per
        item inside the loop leaves items out of the list; the node is still called.
        """
        required_branches: set[RequiredBranch] = set()
        for plan in inputs.values():
            if not plan.sources:
                continue
            behind = [self._branches_required_for(s) for s in plan.sources]
            shared = behind[0].intersection(*behind[1:])
            if isinstance(plan, InputCollectsLoopResults):
                shared = frozenset(branch for branch in shared if branch.depth <= len(plan.loops))
            required_branches |= shared
        return frozenset(required_branches)


def _collects_and_expected_type(field: Field, sources: list[EdgeSource]) -> tuple[bool, Type]:
    """Whether the input ``field`` gathers what ``sources`` bring, and the type each must fit: the input's whole
    type when it reads, one item of a ``Series`` or one record of a table when it gathers.

    A table input reads when every edge brings a table, as one value or as each item of a ``Series`` of tables.
    A value declared ``Any`` says nothing of itself: it is read as a table only where there is nothing to gather,
    made once in the graph, and is taken as a record wherever it is made per item.
    """
    record = field.whole.record
    collects = field.is_series or (record is not None and not all(
        field.whole.fits(s.type) is not None and (s.type.bare is not Any or not s.loops_of_value) for s in sources
    ))
    return collects, field.whole if not collects else field.type if record is None else record


def _bound(fields: Mapping[str, Field], sources: Mapping[str, list[EdgeSource] | None]) -> dict[TypeVar, Any]:
    """What a node's type variables are bound to by what arrives on its edges; two places sharing a variable join.

    Only an edge binds: a typed-in value, a default and an open input bind nothing, and a variable nothing binds
    reads as its bound, else ``Any``.
    """
    bound: dict[TypeVar, Any] = {}
    for name, field in fields.items():
        source_kind = sources.get(name)
        if not field.whole.variables or not source_kind:
            continue
        expected = _collects_and_expected_type(field, source_kind)[1]
        for source in source_kind:
            for variable, hint in (expected.fits(source.type) or {}).items():
                bound[variable] = bound[variable] | hint if variable in bound else hint
    return bound


def _split_gives(whole: Type) -> tuple[Literal["table", "list"], list[str]] | None:
    """What a Split would make of a value of the type ``whole``.

    Returns:
        Which whole value it takes apart, and the fields of each record or item it gives (none for one of unknown
        columns, and for an item that is no record); ``None`` for a type that is neither a table nor a list.
    """
    if whole.record is not None:
        return "table", list(whole.fields)
    one = whole.options[0] if len(whole.options) == 1 else None
    if one is list or typing.get_origin(one) is list:
        return "list", list(Type(next(iter(typing.get_args(one)), Any)).fields)
    return None


def _mismatch(node_id: str, field: Field, source: EdgeSource) -> Problem:
    """The ``type_mismatch`` of an edge from ``source`` into the input ``field``.

    It says the fix where one mistake explains it: a whole table or list wired to an input of one value wants a
    Split first, and a table input fed records without a column it names wants that column.
    """
    address, expected = source.address, field.whole.id
    got = f"Series[{source.type.text}]" if source.item else source.type.text
    split = _split_gives(source.type)
    if field.whole.record is not None:
        has = (source.type.record or source.type).fields
        missing = [column for column in field.whole.fields if column not in has]
        if has and missing:
            column_type = Type(field.whole.fields[missing[0]]).id
            return problem.type_mismatch_add_column(node_id, field.name, source=address, got=got, expected=expected, column=missing[0], column_type=column_type, has=list(has))
    elif split and not field.is_series:
        return problem.type_mismatch_split_first(node_id, field.name, source=address, got=got, expected=expected, whole=split[0], fields=split[1])
    return problem.type_mismatch(node_id, field.name, source=address, got=got, expected=expected)


def _constant_plan(field: Field, constants: Any) -> InputConstant | InputMissing:
    """The plan of an input holding a typed-in value, as its field read it."""
    return InputMissing(field) if constants is _INVALID else InputConstant(field, constants)


def _field_type(record: Type, name: str) -> Type | None:
    """The type of the field ``name`` of a record of the type ``record``.

    Returns:
        A ``TypedDict``'s, dataclass's or pydantic model's declared type for it; a mapping's value type, ``Any``
        for a bare ``dict`` and for a value of any type; ``None`` for a record without the field, and for a type
        that is no record.
    """
    one = record.options[0] if len(record.options) == 1 else None
    if one is Any or one is dict:
        return Type(Any)
    if typing.get_origin(one) in (dict, Mapping):
        return Type(typing.get_args(one)[1])
    return Type(record.fields[name]) if name in record.fields else None


def _hint_and_default(given: Any) -> tuple[Any, Any]:
    """What a hook gives for one field: a hint, or ``(hint, default)`` as pydantic's ``create_model`` takes one."""
    if isinstance(given, tuple) and len(given) == 2:
        return given[0], given[1]
    return given, REQUIRED


def _opposed(a: frozenset[RequiredBranch], b: frozenset[RequiredBranch]) -> bool:
    """Whether two sets of branches never hold together: one decision goes one way for ``a`` and another for ``b``."""
    return any(x[:2] == y[:2] and x.output != y.output for x in a for y in b)


def _reaches(start: str, goal: str, downstream: Mapping[str, list[str]], within: set[str]) -> bool:
    """Whether ``goal`` is read, however far down, by what ``start`` feeds, staying ``within`` those nodes."""
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
