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
  ``"loud/shout.result"``. An exclusive group is named the same way.
- **Item indexes** say which call of a node, when it is called once per item. There is one index per loop the node
  is inside, outermost first. The call of ``loud/shout`` for the line ``" "`` has the item indexes ``(1,)``.
- A **loc** names one call: the node path and the item indexes in one tuple. Each index goes right after the id of
  the node inside its loop, in the graph where the loop is. Here the loop over ``lines.result`` is in the outer
  graph, and the node inside it there is ``loud``. So the call is ``("loud", 1, "shout")``, written
  ``"loud@1/shout"``. A host is given a loc whenever it is told about one call, as in
  ``NotCalled(loc=("loud", 1, "shout"), input="line")`` in ``run.not_called``.
- A **source** is where an edge or a graph output gets its value: one output of a node, or one field of the record
  that output returns. ``Source("lines", "result")`` is written ``"lines.result"``. A graph record stores where
  each edge comes from as a ``Source``. Inside the engine, an output deep in a graph is named by its node path and
  its name instead, as in ``(("loud", "shout"), "result")``.

These texts use ``.``, ``/``, ``@`` and ``~`` as separators. (``~`` starts the fingerprint at the end of a waiting
call's id, as in ``"loud@1/shout~3f2a"``.) So no node id and no name of an input or output may contain any of them.
``is_name`` checks one name, and ``Name`` checks every name in a graph record.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator, ConfigDict, Field, field_validator
from pydantic.dataclasses import dataclass
from pydantic_core import PydanticCustomError

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


#: The pattern of a name: not empty, and none of the characters that places are written with. ``"shout"`` and
#: ``"text-split"`` are names. ``"a.b"`` and ``"loud/shout"`` are not.
_NAME = re.compile(r"^[^./@~]+$")


#: What a name may be, as a problem's message says it.
NAME_RULE = "A name can't be empty, and can't contain '.', '/', '@' or '~'"


def is_name(text: str) -> bool:
    """Whether ``text`` can be a node id or the name of an input or output.

    A name isn't empty and has none of ``.``, ``/``, ``@`` and ``~``, the characters places are written with.
    """
    return _NAME.match(text) is not None


def _a_name(text: str) -> str:
    """Returns ``text`` if it is a name, so that pydantic can use it as a check.

    Raises:
        PydanticCustomError: ``not_a_name`` if it isn't one (see ``is_name``).
    """
    if not is_name(text):
        raise PydanticCustomError(
            "not_a_name",
            "'{text}' isn't a valid name. " + NAME_RULE,
            {"text": text},
        )
    return text


#: A node id, or the name of an input or output, such as ``"shout"`` or ``"text-split"``. Every name in a graph
#: record has this type: node ids, the names of inputs and outputs, and the names of graph inputs and outputs. When
#: a graph record is checked, any other text is refused (see ``is_name``).
Name = Annotated[str, AfterValidator(_a_name)]


@dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid"))
class Source:
    """Where an edge or a graph output gets its value: one output of a node, or one field of the record it returns.

    In a graph record, ``{"node": "lines", "output": "result"}`` is the output ``result`` of the node ``lines``.
    Add ``"field": "email"`` to take only the ``email`` field of the record that output returns. A subgraph node's
    outputs are the graph outputs its graph record declares, and they are named like any other output.

    Every part is checked when a ``Source`` is made, from Python or from JSON. ``Source("loud/shout", "result")``
    raises, since a node id is a name and can't contain ``/``. For a person to read, ``str(source)`` gives
    ``"lines.result"`` or ``"read.result.email"``.

    Attributes:
        node: The node's id.
        output: The output's name.
        field: The field of the record the output returns; ``None`` for the whole output.
    """

    node: Name
    output: Name
    field: Annotated[str | None, Field(exclude_if=lambda field: field is None)] = None

    @field_validator("field")
    @classmethod
    def _a_field(cls, text: str | None) -> str | None:
        """Checks the name of a field of a record: it must not be empty or start with a dot.

        A field's name may have dots elsewhere, as in ``"Amount in USD."``.

        Raises:
            PydanticCustomError: ``not_a_field`` if the name is empty or starts with a dot.
        """
        if text is not None and (not text or text.startswith(".")):
            raise PydanticCustomError(
                "not_a_field",
                "'{text}' isn't a valid field name. A field's name can't be empty or start with '.'",
                {"text": text},
            )
        return text

    def __str__(self) -> str:
        return place_text((self.node,), self.output, self.field)


def place_text(node: NodePath, name: str | None = None, field: str | None = None) -> str:
    """Writes a node, or one of its outputs, as text for a person to read.

    The node path is joined with ``/``. The name of an output or an exclusive group follows after a dot, and a
    record field after another dot. A saved run writes a node the same way.

    ``place_text(("loud", "shout"))`` returns ``"loud/shout"``, ``place_text(("lines",), "result")`` returns
    ``"lines.result"``, and ``place_text(("read",), "result", "email")`` returns ``"read.result.email"``.

    Args:
        node: The node's path.
        name: The name of an output or an exclusive group of the node; ``None`` for the node itself.
        field: The field of the record the output returns; ``None`` for the whole output.
    """
    text = "/".join(node)
    if name is not None:
        text = f"{text}.{name}"
    return text if field is None else f"{text}.{field}"


def parse_node_path(text: str) -> NodePath:
    """Splits a node path that ``place_text`` wrote back into its ids.

    ``parse_node_path("loud/shout")`` returns ``("loud", "shout")``.
    """
    return tuple(text.split("/"))
