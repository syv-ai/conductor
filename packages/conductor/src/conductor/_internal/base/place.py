"""How conductor names a place in a graph or a run: a node, an output, or one call of a node.

Problems, events and saved runs all need to say where something happened. There are four ways to say it, and each
is a tuple for code and a text for people. The examples use this graph, where ``loud`` is a subgraph, a whole graph
used as one node::

    lines(text: str) -> Series[str]   # called once; returns ["a", " ", "c"]
    loud                              # wired to lines.result: its inner nodes are called once per line
        strip(line: str) -> str | None
        shout(line: str) -> str
    join(texts: Series[str]) -> str   # wired to loud.shout.result, the output of loud's inner node shout

- A **node path** names a node. It is the node's id, after the id of each subgraph node it is inside. The inner
  node ``shout`` is ``("loud", "shout")``, written ``"loud/shout"``. The node ``lines`` is ``("lines",)``, written
  ``"lines"``. An output is named by its node path and its name, written with a dot between them:
  ``"loud/shout.result"``.
- **Item indexes** say which call of a node, when it is called once per item. There is one index per loop the node
  is inside, outermost first. The call of ``loud/shout`` for the line ``" "`` has the item indexes ``(1,)``.
- A **loc** names one call: the node path and the item indexes in one tuple. Each index goes right after the id of
  the node inside its loop, in the graph where the loop is. Here the loop over ``lines.result`` is in the outer
  graph, and the node inside it there is ``loud``. So the call is ``("loud", 1, "shout")``, written
  ``"loud@1/shout"``. A host is given a loc whenever it is told about one call, as in
  ``NoneInputCall(loc=("loud", 1, "shout"), input="line")`` in ``run.none_input_calls``.
- A **source** is where an edge or a graph output gets its value: one output of a node, or one field of the value
  that output returns. ``Source("lines", "result")`` is written ``"lines.result"``. A graph record stores where
  each edge comes from as a ``Source``. Inside the engine, an output deep in a graph is named by its node path and
  its name instead, as in ``(("loud", "shout"), "result")``.

These texts use ``.``, ``/``, ``@`` and ``~`` as separators. (``~`` starts the fingerprint at the end of a waiting
call's id, as in ``"loud@1/shout~3f2a"``.) So no node id and no name of an input or output may contain any of them.
``is_name`` checks one name, and ``Name`` checks every name in a graph record. This module writes and splits every
one of these texts, a waiting call's id too, so the separators are named once, below.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator, ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass
from pydantic_core import PydanticCustomError

from conductor._internal.utils import comma_separated

#: The ids on the way to a node, outermost first: the id of each subgraph node it is inside, then its own id.
#: ``("loud", "shout")`` is the node ``shout`` inside the subgraph node ``loud``. ``("lines",)`` is a node of the
#: graph itself. An output of a node is named by its node path and the output's name, side by side.
NodePath = tuple[str, ...]

#: Which call of a node: one index per loop the node is inside, outermost first. In a graph that loops over pages
#: and then over the lines of each page, ``(0, 2)`` is line 2 of page 0. A node inside no loop has ``()``.
ItemIndexes = tuple[int, ...]

#: A node path and its item indexes in one tuple, each index right after the id of the node inside its loop.
#: ``("loud", 1, "shout")`` is the node path ``("loud", "shout")`` with the item indexes ``(1,)``. In
#: ``pages -> lines -> shout``, where ``lines`` returns the lines of each page, ``shout`` is inside two loops of
#: its own graph, so both indexes follow its id: ``("shout", 0, 2)`` is line 2 of page 0.
Loc = tuple[str | int, ...]


#: Comes between a node and the name of its output, and between an output and a field of its value: ``"read.result.email"``.
FIELD_SEPARATOR = "."

#: Comes between the ids of a node path: ``"loud/shout"``.
NODE_SEPARATOR = "/"

#: Comes before each item index in a loc's text: ``"loud@1/shout"``.
INDEX_SEPARATOR = "@"

#: Comes before the fingerprint at the end of a waiting call's id: ``"loud@1/shout~3f2a"``.
FINGERPRINT_SEPARATOR = "~"

#: Every character places are written with. No name may contain one.
SEPARATORS = (FIELD_SEPARATOR, NODE_SEPARATOR, INDEX_SEPARATOR, FINGERPRINT_SEPARATOR)

#: The pattern of a name: not empty, and none of the separators. ``"shout"`` and ``"text-split"`` are names.
#: ``"a.b"`` and ``"loud/shout"`` are not.
_NAME = re.compile(f"^[^{re.escape(''.join(SEPARATORS))}]+$")


def is_name(text: str) -> bool:
    """Whether ``text`` can be a node id or the name of an input or output.

    A name isn't empty and has none of ``.``, ``/``, ``@`` and ``~``, the characters places are written with.
    """
    return _NAME.match(text) is not None


def is_field(text: str) -> bool:
    """Whether ``text`` can be the name of a field of a value that an edge reads.

    A field's name isn't empty and doesn't start with a dot. It may have dots elsewhere, as in ``"Amount in USD."``,
    since a field is the last part of a place's text.
    """
    return bool(text) and not text.startswith(FIELD_SEPARATOR)


def _a_name(text: str) -> str:
    """Returns ``text`` if it is a name, so that pydantic can use it as a check.

    Raises:
        PydanticCustomError: ``not_a_name`` if it isn't one (see ``is_name``).
    """
    if not is_name(text):
        separators = comma_separated(f"'{separator}'" for separator in SEPARATORS)
        raise PydanticCustomError(
            "not_a_name",
            "'{text}' isn't a valid name. A name can't be empty, and can't contain any of {separators}, "
            "since those mean something in sources and waiting ids",
            {"text": text, "separators": separators},
        )
    return text


#: A node id, or the name of an input or output, such as ``"shout"`` or ``"text-split"``. Every name in a graph
#: record has this type: node ids, the names of inputs and outputs, and the names of graph inputs and outputs. When
#: a graph record is checked, any other text is refused (see ``is_name``).
Name = Annotated[str, AfterValidator(_a_name)]


@dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid"))
class Source:
    """Where an edge or a graph output gets its value: one output of a node, or one field of the value it returns.

    In a graph record, ``{"node": "lines", "output": "result"}`` is the output ``result`` of the node ``lines``.
    Add ``"field": "email"`` to take only the ``email`` field of the value that output returns. A subgraph node's
    outputs are the graph outputs its graph record declares, and they are named like any other output.

    Every part is checked when a ``Source`` is made, from Python or from JSON. ``Source("loud/shout", "result")``
    raises, since a node id is a name and can't contain ``/``. For a person to read, ``str(source)`` gives
    ``"lines.result"`` or ``"read.result.email"``.

    Attributes:
        node: The node's id.
        output: The output's name.
        field: The field of the value the output returns; ``None`` for the whole output.
    """

    node: Name
    output: Name
    field: Annotated[str | None, Field(exclude_if=lambda field: field is None)] = None

    @field_validator("field")
    @classmethod
    def _a_field(cls, text: str | None) -> str | None:
        """Checks the name of a field of a value (see ``is_field``).

        Raises:
            PydanticCustomError: ``not_a_field`` if the name is empty or starts with a dot.
        """
        if text is not None and not is_field(text):
            raise PydanticCustomError(
                "not_a_field",
                "'{text}' isn't a valid field name. A field's name can't be empty or start with '.'",
                {"text": text},
            )
        return text

    def __str__(self) -> str:
        return place_text((self.node,), self.output, self.field)


def place_text(loc: Loc, name: str | None = None, field: str | None = None) -> str:
    """Writes a node, one of its inputs or outputs, or one call of a node as text.

    The node ids are joined with ``/``, and each item index follows the id before it after an ``@``. The name of
    an input or an output follows after a dot, and a field of its value after another dot. A node path is a loc with no
    item indexes, and a saved run writes a node this way.

    ``place_text(("loud", "shout"))`` returns ``"loud/shout"``, ``place_text(("lines",), "result")`` returns
    ``"lines.result"``, ``place_text(("read",), "result", "email")`` returns ``"read.result.email"``, and
    ``place_text(("review", 0, "approve", 1))`` returns ``"review@0/approve@1"``.

    Args:
        loc: The node's path, or the loc of one call of it.
        name: The name of an input or an output of the node; ``None`` for the node itself.
        field: The field of the value the output returns; ``None`` for the whole output.
    """
    text = "".join(
        f"{INDEX_SEPARATOR}{part}" if isinstance(part, int) else f"{NODE_SEPARATOR}{part}"
        for part in loc
    )[len(NODE_SEPARATOR) :]
    if name is not None:
        text = f"{text}{FIELD_SEPARATOR}{name}"
    return text if field is None else f"{text}{FIELD_SEPARATOR}{field}"


def parse_node_path(text: str) -> NodePath:
    """Splits a node path that ``place_text`` wrote back into its ids.

    ``parse_node_path("loud/shout")`` returns ``("loud", "shout")``.
    """
    return tuple(text.split(NODE_SEPARATOR))


def parse_loc(text: str) -> Loc:
    """Splits a loc's text back into the loc, as far as it names one.

    ``parse_loc("review@0/approve@1")`` returns ``("review", 0, "approve", 1)``. An item index counts only in the
    spelling ``place_text`` writes: a whole number in the digits 0 to 9, with no sign, space or leading zero, and
    no longer than Python converts to an ``int`` (4300 digits by default). The loc ends at the node
    before the first index that isn't one, so ``"approve@x"`` and ``"approve@01"`` give ``("approve",)``. In the
    same way, the loc ends before the first node id that isn't a name (see ``is_name``), so ``"review//approve"``
    gives ``("review",)`` and ``""`` gives ``()``. To accept only a text ``place_text`` wrote, write the loc back
    and compare: ``place_text(parse_loc(text)) == text``.
    """
    loc: list[str | int] = []
    # the text read so far, to compare with what place_text writes for the loc read so far
    read = ""
    for part in text.split(NODE_SEPARATOR):
        node_id, *indexes = part.split(INDEX_SEPARATOR)
        if not is_name(node_id):
            return tuple(loc)
        read = f"{read}{NODE_SEPARATOR}{node_id}" if loc else node_id
        loc.append(node_id)
        for index in indexes:
            read = f"{read}{INDEX_SEPARATOR}{index}"
            # A text int() can't convert, such as "x" or a number with more digits than Python converts, names no item.
            try:
                with_index = (*loc, int(index))
            except ValueError:
                return tuple(loc)
            # "01", "+1", "1_0" and the Arabic digit "١" convert to a number place_text writes another way.
            if place_text(with_index) != read:
                return tuple(loc)
            loc = list(with_index)
    return tuple(loc)


def write_waiting_id(loc: Loc, fingerprint: str | None) -> str:
    """Writes the id of a waiting call: the call at ``loc``, whose question has the fingerprint ``fingerprint``.

    The id is the loc's text (``place_text``), then ``~`` and the fingerprint. So
    ``write_waiting_id(("review", 0, "approve", 1), "3fa2c0d1")`` returns ``"review@0/approve@1~3fa2c0d1"``. Without
    a fingerprint, the id names just the call: ``"review@0/approve@1"``.
    """
    place = place_text(loc)
    return place if fingerprint is None else f"{place}{FINGERPRINT_SEPARATOR}{fingerprint}"


def parse_waiting_id(text: str) -> tuple[Loc, NodePath, ItemIndexes, str | None]:
    """Splits a waiting id back into the call's loc, its node path, its item indexes and its question's fingerprint,
    as far as it names them.

    ``parse_waiting_id("review@0/approve@1~3fa2c0d1")`` returns the loc ``("review", 0, "approve", 1)``, the node
    path ``("review", "approve")``, the item indexes ``(0, 1)`` and the fingerprint ``"3fa2c0d1"``. The text before
    the ``~`` is split by ``parse_loc``, so ``"approve@x"`` and ``"approve@01"`` give ``("approve",)``, and ``""``
    gives ``()``. With no ``~`` part, or an empty one, the fingerprint is ``None``. To accept only an id
    ``write_waiting_id`` wrote, write it back and compare.
    """
    place, _, fingerprint = text.partition(FINGERPRINT_SEPARATOR)
    loc = parse_loc(place)
    node_path = tuple(part for part in loc if isinstance(part, str))
    item_indexes = tuple(part for part in loc if isinstance(part, int))
    return loc, node_path, item_indexes, fingerprint or None
