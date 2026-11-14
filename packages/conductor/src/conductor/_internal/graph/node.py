"""The nodes of a compiled graph: a ``CompiledFunctionNode`` for a node whose definition is a function, and a
``CompiledSubgraphNode`` for a node whose definition is a subgraph. ``compile.py`` makes them.

The example graph used in this file::

    lines(text: str) -> Series[str]            # called once; returns ["a", "b"]
    shout(line: str, mark: str = "!") -> str   # wired to lines.result: called once per line
    join(texts: Series[str]) -> str            # wired to shout.result: called once, with ["A!", "B!"]

``shout`` compiles to a ``CompiledFunctionNode`` with ``id="shout"``, ``loops=((("lines",), "result"),)``,
a compiled input for each of ``line`` (one item of ``lines.result``) and ``mark`` (the constant ``"!"``), and the
output ``result: str``. A compiled node has nothing from a run in it: it is made once and used by every run.
"""

from __future__ import annotations

import dis
import functools
import inspect
import typing
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import JsonValue

from conductor._internal.ask import Ask, answer_required
from conductor._internal.base import SKIPPED, NodePath, place_text, short_hash
from conductor._internal.field import Field, FieldDescription
from conductor._internal.gate import AnswerDoor, AskDoor, ReturnDoor, undeclared_ask
from conductor._internal.node import FunctionDefinition, SubgraphDefinition
from conductor._internal.series import Series
from conductor._internal.types import Choice, Type

from .input import Input

if TYPE_CHECKING:
    from .type import Graph


class RequiredBranch(NamedTuple):
    """One branch that must be chosen for a node to be called, or for an output to have a value.

    Example: ``route`` returns one of its outputs ``long`` and ``short`` per call, and ``shout`` is wired to
    ``route.long``. ``shout`` is called only where ``route`` chose ``long``:
    ``RequiredBranch(("route",), "length", "long", depth=0)``. Two edges into one input are allowed when what they
    read needs different outputs of one group, since a call then gets a value from at most one of them.

    Attributes:
        node: The path of the node that chooses, from the graph being compiled: ``("route",)``, or ``("e", "route")``
            for a node inside the subgraph node ``e``.
        group: The exclusive group of the outputs it chooses between.
        output: The output of the group that must be the chosen one.
        depth: How many loops the node that chooses is inside: a node inside the loop over ``lines.result``
            chooses once per line.
    """

    node: NodePath
    group: str
    output: str
    depth: int


@dataclass(frozen=True, kw_only=True)
class _CompiledNodeBase:
    """What both kinds of node of a compiled graph have. Never made on its own.

    Attributes:
        id: The node's id in the graph record.
        loops: The loops the node is inside, outermost first, each named by the node path and the output it loops
            over.
        inputs: The compiled input of each input, by name. Each one's ``field`` is the input as compiled: it includes
            the inputs a ``compute_added_inputs`` hook added, and a type variable is replaced by the type its edges
            read.
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
        """Returns the loops ``output`` has one value per item of, outermost first: for a function node, the
        loops the node is inside. In the example, ``shout.loops_around("result")`` is the loop over
        ``lines.result``."""
        return self.loops

    def loop_over(self, output: str) -> tuple[NodePath, str]:
        """Returns the loop over the list ``output`` returns, named as every loop is: by the path of the function
        node that returns the list, and its output. In the example, ``lines.loop_over("result")`` is
        ``(("lines",), "result")``."""
        return (self.id,), output

    def branches_required_for(self, output: str) -> frozenset[RequiredBranch]:
        """Returns the branches that must be chosen for ``output`` to have a value: those the node needs to be
        called, and ``output`` itself when it is in an exclusive group."""
        group = self.outputs[output].param.exclusive_group
        return self.required_branches | {RequiredBranch((self.id,), group, output, len(self.loops))} if group else self.required_branches

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
        code, the types of its inputs and outputs, its loops, and whether it may ask a person.

        A run that reuses the outputs of an earlier run (``with_inputs(previous=run)``) reuses a call only while
        this is the same. So a node redeclared under the same id and version, but with other code, an input or
        output of another type, limit or choice, inside other loops, or with ``asks`` turned on or off, is called
        again: a node that may ask has its outcomes reused once each, in item order, and one that may not without
        limit, so the outcomes of one are never reused as the other's. The code is read from
        the function's own bytecode and constants, not from the functions it calls.
        """
        definition = self.definition
        fields = [(field.name, field.whole.reads_as) for field in (*(node_input.field for node_input in self.inputs.values()), *self.outputs.values())]
        code = _code_text(inspect.unwrap(definition.function).__code__)
        # each loop as place_text writes it (``"e/split.result"``), not as a repr: a waiting call's id and a saved run
        # contain the hash, so it changes only when that text does
        loops = tuple([place_text(node, output) for node, output in self.loops])
        written = repr((definition.id, definition.version, code, fields, loops, definition.asks))
        return short_hash(written.encode(), 8)

    @functools.cached_property
    def return_door(self) -> ReturnDoor:
        """Reads what the node's function returns into its outputs by name: ``{"result": "B!"}`` for ``shout``
        returning ``"B!"``. Built on the node's first call."""
        return ReturnDoor(self.definition.outputs_dataclass, self.definition.compute_outputs is not None, self.outputs)

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
    def questions(self) -> dict[str, FieldDescription]:
        """The question a person is asked about each output, which an answer fills, by name, as the output's type
        sets it: its description, and whether an answer must give it. A field of a returned dataclass with a default
        proposes it. The ``Ask`` can add a prompt, choices and a default to it."""
        return {
            name: output.description.model_copy(update={"required": answer_required(output, {}, limited=output.description.choices is not None)})
            for name, output in self.outputs.items()
        }

    @functools.cached_property
    def ask_door(self) -> Callable[[Ask], tuple[dict[str, tuple[Choice, ...]], dict[str, JsonValue]]]:
        """Checks an ``Ask`` raised for the node against its outputs, which its answer fills, and reads its choices
        and default by output name; for a node that may not ask (``definition.asks``), refuses it (``ask_not_declared``). Built when
        the node first asks.

        Built here, not at ``@node``: a generic node's outputs get their types when the graph is compiled. For
        ``pick[T](value: T) -> T``, ``choices=[1, 2]`` fits when ``value`` comes from an ``int`` output and is refused
        when it comes from a ``str`` one."""
        return AskDoor(self.outputs) if self.definition.asks else undeclared_ask

    @functools.cached_property
    def answer_door(self) -> AnswerDoor:
        """Reads a person's answer to a waiting call of the node into its outputs: ``{"result": True}`` for
        ``approve`` answered ``True``. Built when the node is first answered."""
        return AnswerDoor(self.outputs, self.questions)


@typing.final
@dataclass(frozen=True, kw_only=True)
class CompiledSubgraphNode(_CompiledNodeBase):
    """A node of a compiled graph whose definition is a subgraph. The subgraph's own graph record is compiled
    on its own, as ``inner``, and the run calls the function nodes inside it in this node's loops.

    Its inputs and outputs are the graph inputs and outputs the subgraph's record declares, by name. It has no policy (retries, threads, timeouts),
    and its outputs are never reused as a whole: each node inside is called by its own policy, and a later run
    reuses the outputs of the nodes inside one by one, as it would if they were drawn in the graph itself.

    An output is read outside as it would be if the subgraph's nodes were drawn in the outer graph. Example: the
    subgraph ``documents -> lines -> shout``, used as the node ``e``, whose record declares the graph output
    ``shouted``, the output ``result`` of ``shout``. ``lines`` is called once per document and ``shout`` once per
    line. So a node outside wired to ``e.shouted`` that takes one ``str`` is called once
    per line of each document, and one that takes a ``Series[str]`` once per document. ``outputs`` has the output
    as ``shout`` declares it (``str``), and ``loops_around`` names the two loops. Its description says
    ``Series[Series[str]]``, one ``Series`` per loop.

    Attributes:
        definition: The node definition, as its author declared it with ``subgraph(...)``.
        inner: The subgraph's graph record, compiled on its own. It is never run on its own.
    """

    definition: SubgraphDefinition
    inner: Graph[Any]

    def loops_around(self, output: str) -> tuple[tuple[NodePath, str], ...]:
        """Returns the loops ``output`` has one value per item of, outermost first: the loops this node is
        inside, then the loops inside the subgraph.

        Example: ``e`` above is called once, and ``loops_around("shouted")`` is
        ``((("e", "documents"), "result"), (("e", "lines"), "result"))``. A loop inside is named by the path of the node
        inside, from this graph, and its output, so two outputs returned in the same loop inside name the same loop,
        and a node outside that reads both is called once per item with the two values of that item.
        """
        inside = self.inner._loops_around(output)
        return (*self.loops, *[((self.id, *node), name) for node, name in inside])

    def loop_over(self, output: str) -> tuple[NodePath, str]:
        """Returns the loop over the list ``output`` returns, named by the function node inside that returns it.

        Example: the subgraph ``s`` is ``chars -> upper``, and its record declares ``chars.result`` as the graph
        output ``parts``. ``s.loop_over("parts")`` is ``(("s", "chars"), "result")``, the loop ``loops_around`` names
        for ``upper``'s output, so a node outside that reads one item of ``parts`` and the value of ``upper`` for
        that item is called once per item with both.
        """
        source = self.inner._outputs[output]
        node, name = self.inner._nodes[source.node[0]].loop_over(source.output)
        return (self.id, *node), name

    def branches_required_for(self, output: str) -> frozenset[RequiredBranch]:
        """Returns the branches that must be chosen for ``output`` to have a value: those this node needs to be
        called, and those the output needs inside the subgraph.

        Example: the subgraph ``e`` is ``route -> shout`` and ``route -> whisper``, one on each output of
        ``route``, and its record declares the graph outputs ``loud`` (``shout.result``) and ``quiet``
        (``whisper.result``). ``branches_required_for("loud")`` is ``RequiredBranch(("e", "route"), "length", "long",
        0)`` and that of ``quiet`` names ``short``, so one input outside may be wired to both, as it may with the same
        nodes drawn in the outer graph. A branch inside is named by the path of its node, from this graph.
        """
        inside = self.inner._branches_required_for(output)
        depth = len(self.loops)
        return self.required_branches | {
            RequiredBranch((self.id, *branch.node), branch.group, branch.output, depth + branch.depth)
            for branch in inside
        }

    def described_outputs(self) -> list[FieldDescription]:
        """Returns each output as an editor shows it: as a list with one level per loop inside the subgraph, with
        those loops by name.

        Example: ``shouted`` of ``e`` above is described with ``is_series`` true, the item type
        ``Series[str]`` (so the whole output is ``Series[Series[str]]``), and the ``loops``
        ``["documents.result", "lines.result"]``.
        """
        described = []
        series_of: Any = Series  # typed Any: the type checker cannot follow ``Series[...]`` built at run time
        for name, field in self.outputs.items():
            loops = [place_text(node[1:], output) for node, output in self.loops_around(name)[len(self.loops) :]]
            if not loops:
                described.append(field.description)
                continue
            if field.is_series:  # an output that is itself a Series adds the innermost level, the list it is inside
                node, output = self.loop_over(name)
                loops.append(place_text(node[1:], output))
            item: Any = field.type.hint
            for _ in loops[1:]:
                item = series_of[item]
            item_type = Type(item)
            series_field = field if field.is_series else Field(name, series_of[field.hint])
            described.append(series_field.description.model_copy(update={"type": item_type.id, "json_schema": item_type.json_schema, "loops": loops}))
        return described


def outputs_when_skipped(outputs: Mapping[str, Field]) -> dict[str, Any]:
    """The outputs of a call that was skipped, of a node with these outputs: ``SKIPPED`` for each.

    For ``shout``: ``{"result": SKIPPED}``.
    """
    return dict.fromkeys(outputs, SKIPPED)


def outputs_when_not_called(outputs: Mapping[str, Field]) -> dict[str, Any]:
    """The outputs of a call that was not made because an argument was ``None``, of a node with these outputs.

    For ``shout``: ``{"result": None}``. For ``route``, whose two outputs are branches, and a node with one more
    output ``note`` that is not: ``{"short": SKIPPED, "long": SKIPPED, "note": None}``. An output is a branch
    when its ``Param`` gives it an ``exclusive_group``.
    """
    return {name: SKIPPED if output.param.exclusive_group is not None else None for name, output in outputs.items()}


#: One node of a compiled graph: a function, or a subgraph.
CompiledNode = CompiledFunctionNode | CompiledSubgraphNode


#: The code flag that says a function has a docstring, from Python 3.14; ``None`` before.
_HAS_DOCSTRING_FLAG: int | None = getattr(inspect, "CO_HAS_DOCSTRING", None)


def _code_text(code: Any) -> str:
    """Returns a function's bytecode, constants and names as text that is the same in every process, without its
    docstrings. A function or class defined inside it is written by its own code.

    A docstring does not change what a node does, so adding, changing or removing one keeps the text. A function's
    docstring is its first constant, so it is written as ``None``: Python 3.12 and 3.13 keep that place for it
    (``None`` when there is none), and from 3.14 a flag says whether there is one. A class body sets ``__doc__`` with two instructions of its own, so it is written
    instruction by instruction without them.
    """
    if not code.co_flags & inspect.CO_NEWLOCALS:
        instructions = list(dis.get_instructions(code))
        kept = [
            instruction for index, instruction in enumerate(instructions)
            if not (instruction.opname == "STORE_NAME" and instruction.argval == "__doc__")
            and not (instruction.opname == "LOAD_CONST" and index + 1 < len(instructions) and instructions[index + 1].argval == "__doc__")
        ]
        return f"{[(i.opname, _constant_text(i.argval) if i.opname == 'LOAD_CONST' else repr(i.argval)) for i in kept]}"
    constants = list(code.co_consts)
    if _HAS_DOCSTRING_FLAG is None:
        has_docstring = bool(constants) and isinstance(constants[0], str) and code.co_name != "<genexpr>"
    else:
        has_docstring = bool(code.co_flags & _HAS_DOCSTRING_FLAG)
    if has_docstring:
        constants[0] = None
    return f"{code.co_code.hex()}:{[_constant_text(c) for c in constants]}:{code.co_names}"


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
