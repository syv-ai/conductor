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
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from conductor._internal.ask import AskValidator
from conductor._internal.base import SKIPPED, NodePath, place_text, short_hash
from conductor._internal.field import Field, FieldDescription
from conductor._internal.node import FunctionDefinition, ReturnValidator, SubgraphDefinition

from .input import Input

if TYPE_CHECKING:
    from .type import Graph


class RequiredBranch(NamedTuple):
    """A branch that must be chosen for a node to be called, or for an output to have a value.

    Some nodes choose. Each call fills exactly one output of an exclusive group, and the others are skipped. Say
    ``route`` has the outputs ``long`` and ``short`` in the group ``length``, and ``shout`` is wired to
    ``route.long``. Then ``shout`` is called only where ``route`` chose ``long``, so it needs
    ``RequiredBranch(("route",), "length", "long", depth=0)``.

    The compiler uses these to allow two edges into one input. That's fine when the two need different outputs of
    one group, since a call then gets a value from at most one of them.

    Attributes:
        node: The path of the node that chooses, from the graph being compiled. That's ``("route",)``, or
            ``("e", "route")`` for a node inside the subgraph node ``e``.
        group: The exclusive group the node chooses an output of.
        output: The output of the group that must be chosen.
        depth: How many loops the choosing node is in. A node inside the loop over ``lines.result`` chooses once
            per line.
    """

    node: NodePath
    group: str
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
        required_branches: The branches that must be chosen upstream for the node to be called.
    """

    id: str
    loops: tuple[tuple[NodePath, str], ...]
    inputs: dict[str, Input]
    outputs: dict[str, Field]
    required_branches: frozenset[RequiredBranch]

    def loops_around(self, output: str) -> tuple[tuple[NodePath, str], ...]:
        """Returns the loops ``output`` has one value per item of, outermost first.

        For a function node, these are just the loops the node is in. In the example,
        ``shout.loops_around("result")`` is the loop over ``lines.result``.
        """
        return self.loops

    def returned_by(self, output: str) -> tuple[NodePath, str]:
        """Returns the function node that returns ``output``, as a path from this graph, and the output's name there.

        For a function node, that's the node itself and ``output``. In the example, ``lines.returned_by("result")``
        is ``(("lines",), "result")``. That's also how the loop over that list is named.
        """
        return (self.id,), output

    def branches_required_for(self, output: str) -> frozenset[RequiredBranch]:
        """Returns the branches that must be chosen for ``output`` to have a value.

        These are the branches the node needs to be called at all. If ``output`` is in an exclusive group, it's
        added too, since it has a value only where the node chose it.
        """
        group = self.outputs[output].param.exclusive_group
        return (
            self.required_branches | {RequiredBranch((self.id,), group, output, len(self.loops))}
            if group
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

    @functools.cached_property
    def code_fingerprint(self) -> str:
        """A short hash of what a call's outputs depend on, besides its arguments.

        That's the node's id and version, its code, the types of its inputs and outputs, its loops, and whether it
        may ask a person.

        A run can reuse calls from an earlier run (``with_inputs(previous=run)``), but only while this hash is the
        same. So the node is called again if it's redeclared under the same id and version with other code, an
        input or output of another type, limit or choice, other loops, or ``asks`` turned on or off.

        ``asks`` is in the hash because the two kinds of node are reused differently. A node that may ask has each
        outcome reused once, in item order. One that may not has its outcomes reused any number of times. So the
        outcomes of one are never reused as the other's.

        The code is the function's own parsed source, not the functions it calls, so a new Python version keeps the
        hash. A function whose source can't be read, such as a lambda or one made by ``exec``, is hashed by its
        bytecode instead (see ``code_text_of``).
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
        loops = tuple([place_text(node, output) for node, output in self.loops])
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
        """The outputs of the node when it isn't called because it's on a branch that wasn't chosen.

        Every output is ``SKIPPED``. For ``shout``, that's ``{"result": SKIPPED}``.
        """
        return dict.fromkeys(self.outputs, SKIPPED)

    @functools.cached_property
    def outputs_when_not_called(self) -> dict[str, Any]:
        """The outputs of the node when it isn't called because an argument was ``None``, and its parameter doesn't
        take ``None``.

        An output that's a branch (``Field.is_branch``) is ``SKIPPED``, since it wasn't chosen. Every other output
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

    ``outputs`` has the output as ``shout`` declares it (``str``), and ``loops_around`` names the two loops. Its
    description says it's a list (``is_series``) of ``str``, with one level of list per loop.

    Attributes:
        definition: The node definition, as its author declared it with ``subgraph(...)``.
        inner: The subgraph's graph record, compiled on its own. It's never run on its own.
    """

    definition: SubgraphDefinition
    inner: Graph[Any]

    def loops_around(self, output: str) -> tuple[tuple[NodePath, str], ...]:
        """Returns the loops ``output`` has one value per item of, outermost first.

        These are the loops this node is in, followed by the loops inside the subgraph. For ``e`` above, which is
        called once, ``loops_around("shouted")`` is ``((("e", "documents"), "result"), (("e", "lines"), "result"))``.

        A loop inside is named by the path of the node inside, from this graph, and its output. So two outputs
        returned in the same loop inside name the same loop. A node outside that reads both is then called once per
        item, with both values of that item.
        """
        source, output_inside = self.inner._node_returning(output)
        inside = source.loops_around(output_inside)
        return (*self.loops, *[((self.id, *node), name) for node, name in inside])

    def returned_by(self, output: str) -> tuple[NodePath, str]:
        """Returns the function node inside the subgraph that returns ``output``, as a path from this graph, and the
        output's name there.

        Say the subgraph ``s`` is ``chars -> upper``, and its record declares ``chars.result`` as the graph output
        ``parts``. Then ``s.returned_by("parts")`` is ``(("s", "chars"), "result")``. That's also how
        ``loops_around`` names the loop over ``parts`` for ``upper``'s output. So a node outside that reads one item
        of ``parts``, and ``upper``'s value for that item, is called once per item with both.
        """
        source, output_inside = self.inner._node_returning(output)
        path, returned = source.returned_by(output_inside)
        return (self.id, *path), returned

    def branches_required_for(self, output: str) -> frozenset[RequiredBranch]:
        """Returns the branches that must be chosen for ``output`` to have a value.

        These are the branches this node needs to be called, and those the output needs inside the subgraph. A
        branch inside is named by the path of its node, from this graph.

        Say the subgraph ``e`` is ``route -> shout`` and ``route -> whisper``, one on each output of ``route``. Its
        record declares the graph outputs ``loud`` (``shout.result``) and ``quiet`` (``whisper.result``). Then
        ``branches_required_for("loud")`` is ``RequiredBranch(("e", "route"), "length", "long", 0)``, and the one
        for ``quiet`` names ``short``. So one input outside may be wired to both, just as it could if the nodes were
        drawn in the outer graph.
        """
        source, output_inside = self.inner._node_returning(output)
        inside = source.branches_required_for(output_inside)
        depth = len(self.loops)
        return self.required_branches | {
            RequiredBranch(
                (self.id, *branch.node), branch.group, branch.output, depth + branch.depth
            )
            for branch in inside
        }

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
            loops = [
                place_text(node[1:], output)
                for node, output in self.loops_around(name)[len(self.loops) :]
            ]
            if not loops:
                described.append(field.description)
                continue
            # an output that's a Series itself adds one more, innermost level: the list it returns
            if field.is_series:
                node, output = self.returned_by(name)
                loops.append(place_text(node[1:], output))
            described.append(
                field.description.model_copy(update={"is_series": True, "loops": loops})
            )
        return described


#: One node of a compiled graph: a function, or a subgraph.
CompiledNode = CompiledFunctionNode | CompiledSubgraphNode
