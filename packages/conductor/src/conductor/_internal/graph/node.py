"""The nodes of a compiled graph.

There are two kinds. A ``CompiledFunctionNode`` is a node whose definition is a function, made with ``@node``. A
``CompiledSubgraphNode`` is a node whose definition is a whole other graph, made with ``subgraph()``. The compiler
in ``compile.py`` makes both.

The example graph used in this file::

    lines(text: str) -> Series[str]            # called once; returns ["a", "b"]
    shout(line: str, mark: str = "!") -> str   # wired to lines.result: called once per line
    join(texts: Series[str]) -> str            # wired to shout.result: called once, with ["A!", "B!"]

``shout`` compiles to a ``CompiledFunctionNode`` with ``id="shout"`` and ``loops=((("lines",), "result"),)``,
since it's called once per line. It has a compiled input for ``line`` (one item of ``lines.result``) and one for
``mark`` (the constant ``"!"``), and the output ``result: str``.

Nothing from a run is stored on a compiled node. It's made once, and every run uses it.
"""

from __future__ import annotations

import functools
import typing
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, NamedTuple

from conductor._internal.ask import AskValidator
from conductor._internal.base import SKIPPED, NodePath, Source, place_text, short_hash
from conductor._internal.field import Field, FieldDescription
from conductor._internal.node import FunctionDefinition, ReturnValidator, SubgraphDefinition

from .input import Edge, Input

if TYPE_CHECKING:
    from .type import Graph


class RequiredBranch(NamedTuple):
    """A branch that must be taken for a node to be called, or for an output to have a value.

    A branch is an output whose default is ``SKIPPED``, and a call gives a value to at most one branch of a node.
    Say ``route`` has the branches ``long`` and ``short``, and ``shout`` is wired to ``route.long``. Then ``shout``
    is called only where ``route`` gave ``long`` a value, so it needs ``RequiredBranch("route", "long", depth=0)``.

    The compiler uses these to allow two edges into one input. That's fine when the two need different branches of
    one node, since a call then gets a value from at most one of them.

    Attributes:
        node: The id of the function node the branch belongs to. A subgraph node has no branches.
        output: The branch that must be taken.
        depth: How many loops that node is in. A node inside the loop over ``lines.result`` takes a branch once
            per line.
    """

    node: str
    output: str
    depth: int


@dataclass(frozen=True, kw_only=True)
class _CompiledNodeBase:
    """What both kinds of compiled node have. It's never made on its own.

    Attributes:
        id: The node's id in the graph record.
        loops: The loops the node is in, outermost first. Each is named by the node path and output of the list it
            loops over.
        inputs: The compiled input of each input, by name. Each one's ``field`` is the input as compiled. Inputs a
            ``compute_added_inputs`` hook added are included, and a type variable is replaced by the type its
            edges carry.
        outputs: Each output, by name, as compiled: as the definition declares it, or as a ``compute_outputs``
            hook returned it.
        required_branches: The branches that must be taken upstream for the node to be called.
    """

    id: str
    loops: tuple[tuple[NodePath, str], ...]
    inputs: dict[str, Input]
    outputs: dict[str, Field]
    required_branches: frozenset[RequiredBranch]

    def edge_reading(self, output: str) -> Edge:
        """Returns the edge that reads the whole of ``output``.

        Its ``node`` and ``output`` name the function node that returns the value, and the output there. Its
        ``loops`` are the loops that function node is in. For a function node, that's the node itself and its own
        loops. In the example, ``shout.edge_reading("result")`` has the ``node`` ``("shout",)`` and the loop over
        ``lines.result``.

        The ``node`` and ``output`` of ``lines.edge_reading("result")`` are ``("lines",)`` and ``"result"``. That's
        also how the loop over that list is named.
        """
        field = self.outputs[output]
        return Edge(
            Source(self.id, output),
            (self.id,),
            output,
            self.loops,
            field.type,
            per_item=field.is_series,
        )

    def branches_required_for(self, output: str) -> frozenset[RequiredBranch]:
        """Returns the branches that must be taken for ``output`` to have a value.

        These are the branches the node needs to be called at all. If ``output`` is a branch, it's added too, since
        it has a value only where the node gave it one.
        """
        return (
            self.required_branches | {RequiredBranch(self.id, output, len(self.loops))}
            if self.outputs[output].is_branch
            else self.required_branches
        )

    def described_outputs(self) -> list[FieldDescription]:
        """Returns each output as an editor shows it.

        A waiting call shows its questions this way too, when its ``Ask`` sets no choices or default.
        """
        return [field.description for field in self.outputs.values()]


@typing.final
@dataclass(frozen=True, kw_only=True)
class CompiledFunctionNode(_CompiledNodeBase):
    """A compiled node whose definition is a function, like ``shout`` in the example.

    Attributes:
        definition: The node definition, as its author declared it with ``@node``.
    """

    definition: FunctionDefinition[..., Any]

    def code_fingerprint(self, outer_loops: tuple[tuple[NodePath, str], ...]) -> str:
        """Returns a short hash of what a call's outputs depend on, besides its arguments.

        That's the node's ``@node`` id and version, its code, the name and type of each input and output (limits and
        choices included), the loops it runs in, and whether it may ask a person.

        The loops it runs in are its own loops and, inside a subgraph, the loops the subgraph node is in. One
        compiled subgraph may be used in several places, so each blueprint node works its hash out with the loops
        of its own place (``BlueprintFunctionNode.code_fingerprint``). Moving a subgraph node into a loop changes
        the hash of every node inside it, as moving the nodes themselves into the loop would.

        The loops around the subgraph are written by their path in the outer graph, and the node's own loops by
        their path in its own graph, with ``"/"`` between the two. So a loop moved from around the subgraph node to
        inside its graph changes the hash too, even when the node that returns the list keeps its id. A node with
        no loops around its subgraph writes its own loops alone, so its hash is the same in every place it's used.

        A run can reuse calls from an earlier run (``with_inputs(previous=run)``), but only while this hash is the
        same. So the node is called again if it's redeclared under the same id and version with other code, an
        input or output of another type, limit or choice, other loops, or ``asks`` turned on or off.

        ``asks`` is in the hash because the two kinds of node are reused differently. A node that may ask has each
        outcome reused once, in item order. One that may not has its outcomes reused any number of times. So the
        outcomes of one are never reused as the other's.

        The code is the function's own parsed source, not the functions it calls, so a new Python version keeps the
        hash. A function whose source can't be read, such as a lambda or one made by ``exec``, is hashed by its
        bytecode instead (see ``code_text_of``).

        Args:
            outer_loops: The loops around the subgraph this node is in, named from the outer graph. Empty for a node
                of the outer graph.
        """
        definition = self.definition
        fields = [
            (field.name, field.whole.reads_as)
            for field in (
                *(node_input.field for node_input in self.inputs.values()),
                *self.outputs.values(),
            )
        ]
        code = definition.code_text
        # Each loop is written with place_text, like "e/split.result", not with repr. Waiting-call ids and saved
        # runs contain the hash, so it should change only when that text does.
        own = tuple([place_text(node, output) for node, output in self.loops])
        around = tuple([place_text(node, output) for node, output in outer_loops])
        loops = (*around, "/", *own) if around else own
        written = repr((definition.id, definition.version, code, fields, loops, definition.asks))
        return short_hash(written.encode(), 8)

    @functools.cached_property
    def return_validator(self) -> ReturnValidator:
        """The validator that turns what the node's function returns into its outputs by name.

        For ``shout`` returning ``"B!"``, it gives ``{"result": "B!"}``. Built on the node's first call.
        """
        return ReturnValidator(
            self.definition.outputs_dataclass,
            self.definition.compute_outputs is not None,
            self.outputs,
        )

    @functools.cached_property
    def outputs_when_skipped(self) -> dict[str, Any]:
        """The outputs of the node when it isn't called because it's on a skipped branch.

        Every output is ``SKIPPED``. For ``shout``, that's ``{"result": SKIPPED}``.
        """
        return dict.fromkeys(self.outputs, SKIPPED)

    @functools.cached_property
    def outputs_when_not_called(self) -> dict[str, Any]:
        """The outputs of the node when it isn't called because an argument was ``None``, and its parameter doesn't
        take ``None``.

        An output that's a branch (``Field.is_branch``) is ``SKIPPED``, since it wasn't taken. Every other output
        is ``None``. For ``shout``, that's ``{"result": None}``. Say ``route`` has the branches ``short`` and
        ``long``, and one more output ``note`` that isn't a branch. Then it's
        ``{"short": SKIPPED, "long": SKIPPED, "note": None}``.
        """
        return {
            name: SKIPPED if output.is_branch else None for name, output in self.outputs.items()
        }

    @functools.cached_property
    def ask_validator(self) -> AskValidator:
        """The validator for an ``Ask`` the node raises. Built when the node first asks.

        It checks the ``Ask``'s choices and default against the node's outputs, which the answer fills. If the node
        doesn't say ``asks=True``, every ``Ask`` is refused with ``ask_not_declared``.

        It's built here, after compiling, and not at ``@node``. That's because a generic node's outputs get their
        types when the graph is compiled. For ``pick[T](value: T) -> T``, ``choices=[1, 2]`` fits when ``value``
        comes from an ``int`` output, and is refused when it comes from a ``str`` one.
        """
        return AskValidator(self.outputs, asks=self.definition.asks)


@typing.final
@dataclass(frozen=True, kw_only=True)
class CompiledSubgraphNode(_CompiledNodeBase):
    """A compiled node whose definition is a subgraph.

    The subgraph's own record is compiled on its own, as ``inner``. When the graph runs, the function nodes inside
    it are called within this node's loops.

    The node's inputs and outputs are the graph inputs and outputs the subgraph's record declares, by name. It has
    no policy of its own (no retries, threads or timeouts), and its outputs are never reused as a whole. Each node
    inside follows its own policy, and a later run reuses the outputs of the nodes inside one by one. Both work just
    as they would if the nodes were drawn in the graph itself.

    The same goes for how a node outside gets its outputs. Take the subgraph ``documents -> lines -> shout``, used
    as the node ``e``. Its record declares the graph output ``shouted`` as ``shout.result``. ``lines`` is called
    once per document, and ``shout`` once per line. So a node outside wired to ``e.shouted`` is called once per line
    of each document if it takes one ``str``, and once per document if it takes a ``Series[str]``.

    ``outputs`` has the output as ``shout`` declares it (``str``), and ``edge_reading`` names the two loops. Its
    description says it's a list (``is_series``) of ``str``, with one level of list per loop.

    Attributes:
        definition: The node definition, as its author declared it with ``subgraph(...)``.
        inner: The subgraph's graph record, compiled on its own. It's never run on its own.
    """

    definition: SubgraphDefinition
    inner: Graph[Any]

    def edge_reading(self, output: str) -> Edge:
        """Returns the edge that reads the whole of ``output``.

        It's the edge that reads the graph output inside the subgraph, named from this graph
        (``Edge.in_outer_graph``). For ``e`` above, which is called once, ``edge_reading("shouted")`` has the
        ``node`` ``("e", "shout")``. Its ``loops`` are the loop over ``e/documents.result``, then the loop over
        ``e/lines.result``.

        A loop inside is named by the path of the node inside, from this graph, and its output. So two outputs
        returned in the same loop inside name the same loop. A node outside that reads both is then called once per
        item, with both values of that item.

        Say the subgraph ``s`` is ``chars -> upper``, and its record declares ``chars.result`` as the graph output
        ``parts``. Then ``s.edge_reading("parts")`` has the ``node`` ``("s", "chars")`` and the ``output``
        ``"result"``. That's also how ``upper``'s output names the loop over ``parts``.
        """
        edge = self.inner._edge_reading(output).in_outer_graph((self.id,), self.loops)
        return replace(edge, wired_to=Source(self.id, output), type=self.outputs[output].type)

    def described_outputs(self) -> list[FieldDescription]:
        """Returns each output as an editor shows it.

        An output is described as the output of the node inside that returns it. If that node is inside loops in
        the subgraph, the output is described as a list (``is_series``), with one level per entry of ``loops``.

        For ``e`` above, ``shouted`` is described as ``shout``'s ``result``: the type ``str``, ``is_series`` true,
        and the ``loops`` ``["documents.result", "lines.result"]``. That's a list per document of a list per line
        of ``str``. If ``shout`` returned a ``Series[str]``, ``loops`` would end with ``"shout.result"`` too, a
        third level.
        """
        described = []
        for name, field in self.outputs.items():
            # The edge as the subgraph's own graph names it, since the description names the loops that way.
            inner_edge = self.inner._edge_reading(name)
            loops = [place_text(*loop) for loop in inner_edge.loops]
            if not loops:
                described.append(field.description)
                continue
            # an output that's a Series itself adds one more, innermost level: the list it returns
            if field.is_series:
                loops.append(place_text(inner_edge.node, inner_edge.output))
            described.append(
                field.description.model_copy(update={"is_series": True, "loops": loops})
            )
        return described


#: One node of a compiled graph: a function, or a subgraph.
CompiledNode = CompiledFunctionNode | CompiledSubgraphNode
