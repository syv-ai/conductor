"""Four ways to name a place in a graph or a run, and how each is written as text.

The example graph is the README's, ``lines -> shout -> join``, with ``shout`` replaced by a subgraph: a graph
registered as a node definition. The subgraph ``loud`` is the graph ``strip -> shout``, used as one node::

    lines(text: str) -> Series[str]   # called once; returns ["a", " ", "c"]
    loud                              # wired to lines.result: its inner nodes are called once per line
        strip(line: str) -> str | None
        shout(line: str) -> str
    join(texts: Series[str]) -> str   # wired to loud.shout.result, the output of loud's inner node shout

- A **node path** names a node: its id, after the id of each subgraph node it is inside. The inner node ``shout``
  is ``("loud", "shout")``, written ``"loud/shout"``; ``lines`` is ``("lines",)``, written ``"lines"``.
- **Item indexes** say which call of a node: one index per loop the node is inside, outermost first. The call of
  ``loud/shout`` for the line ``" "`` has the item indexes ``(1,)``.
- A **loc** names one call: the node path and the item indexes in one tuple. Each index is written after the id
  of the node that is inside its loop in the graph where the loop is. That call is ``("loud", 1, "shout")``,
  written ``"loud@1/shout"``: the loop over ``lines.result`` is in the outer graph, where the node inside it is
  ``loud``, so the index follows ``"loud"``. A host is given a loc when it
  is told about one call, as in ``run.not_called``: ``NotCalled(loc=("loud", 1, "shout"), input="line")``.
- An **address** names one output of a node, or one field of the record an output returns: ``Address("lines",
  "result")``, written ``"lines.result"``. A graph record writes where each edge comes from as one; compiling names a
  loop by the address of the list it loops over.

Because these are written with ``.``, ``/``, ``@`` and ``~`` (``~`` comes before the fingerprint of a waiting
call's id, ``"loud@1/shout~3f2a"``), no node id and no name of an input or output contains one of them (``is_name``).
``Name`` and the fields of ``Address`` hold a graph record to that rule where it is read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated

from pydantic import AfterValidator, Field
from pydantic_core import PydanticCustomError

#: The names on the way to a place in a graph, outermost first: the id of each subgraph node it is inside, the node's
#: own id, and for an output or an exclusive group of the node, its name. ``("loud", "shout")`` is the node ``shout``
#: inside the subgraph node ``loud``, and ``("loud", "shout", "result")`` its output ``result``; ``("lines",)`` is a node
#: of the graph itself, and ``("lines", "result")`` its output. Compiling names each loop by the path of the list it
#: loops over, and each branch by the path of the node that chooses it.
Path = tuple[str, ...]

#: Which call of a node: one index per loop the node is inside, outermost first. In a graph that loops over pages
#: and then over the lines of each page, ``(0, 2)`` is line 2 of page 0. A node inside no loop has ``()``.
ItemIndexes = tuple[int, ...]

#: A node path and item indexes in one tuple, each index after the id of the node inside its loop.
#: ``("loud", 1, "shout")`` is the node path ``("loud", "shout")`` with the item indexes ``(1,)``. In
#: ``pages -> lines -> shout``, where ``lines`` returns the lines of each page, ``shout`` is inside two loops of
#: its own graph, so both indexes follow its id: ``("shout", 0, 2)`` is line 2 of page 0.
Loc = tuple[str | int, ...]


#: A node id, or the name of an input or output: not empty, and none of the characters addresses, node paths, locs
#: and waiting ids are written with. ``"shout"`` and ``"text-split"`` are names; ``"a.b"`` and ``"loud/shout"`` are not.
_NAME = re.compile(r"^[^./@~]+$")


def is_name(text: str) -> bool:
    """Whether ``text`` can be a node id or the name of an input or output : not empty, and none of ``.``, ``/``, ``@`` and ``~``."""
    return _NAME.match(text) is not None


def _a_name(text: str) -> str:
    """Returns ``text`` when it is a name (``is_name``).

    Raises:
        PydanticCustomError: It is not one (``not_a_name``).
    """
    if not is_name(text):
        raise PydanticCustomError("not_a_name", "'{text}' is not a name: a name is not empty and has none of '.', '/', '@' and '~'", {"text": text})
    return text


def _a_field(text: str | None) -> str | None:
    """Returns ``text``, the name of a field of a record, when it is not empty and does not start with a dot. A
    field's name may have dots of its own: ``"Amount in USD."``.

    Raises:
        PydanticCustomError: It is empty or starts with a dot (``not_a_field``).
    """
    if text is not None and (not text or text.startswith(".")):
        raise PydanticCustomError("not_a_field", "'{text}' is not the name of a field: it is not empty and does not start with '.'", {"text": text})
    return text


#: A node id, or the name of an input or output: ``"shout"``, ``"text-split"``. Every name in a graph record is one:
#: a node's id, an input's, an output's and a graph input's or output's. Read from a graph record, anything else is
#: refused (``is_name``).
Name = Annotated[str, AfterValidator(_a_name)]

@dataclass(frozen=True, slots=True)
class Address:
    """One output of a node, or one field of the record an output returns: where an edge comes from.

    A graph record writes it as ``{"node": "lines", "output": "result"}``, and a field as ``{"node": "read",
    "output": "result", "field": "email"}``. A subgraph node's outputs are the graph outputs its graph record
    declares, each a name like any other output's.

    Compiling names each loop by the address of the list it loops over, and the run's flat graph names a node by
    its node path as text (``Address("loud/shout", "result")``). As text, for a person to read, the parts are
    joined by dots: ``"lines.result"``, ``"read.result.email"``.

    Attributes:
        node: The node's id; in the flat graph, its node path as text.
        output: The output's name.
        field: The field of the record the output returns; ``None`` for the whole output.
    """

    node: Name
    output: Name
    field: Annotated[str | None, AfterValidator(_a_field), Field(exclude_if=lambda field: field is None)] = None

    def __str__(self) -> str:
        return f"{self.node}.{self.output}" if self.field is None else f"{self.node}.{self.output}.{self.field}"


def loc_text(loc: Loc) -> str:
    """A loc as text: ``/`` between node ids and ``@`` before each index.

    ``loc_text(("loud", 1, "shout"))`` returns ``"loud@1/shout"``, and ``loc_text(("shout", 0, 2))`` returns
    ``"shout@0@2"``. A loc starts with a node id.
    """
    return "".join(f"@{part}" if isinstance(part, int) else f"/{part}" for part in loc)[1:]


def parse_loc_text(text: str) -> tuple[Loc, Path, ItemIndexes]:
    """Read the text of a loc back, and also split it into its node path and its item indexes.

    ``parse_loc_text("loud@1/shout")`` returns ``(("loud", 1, "shout"), ("loud", "shout"), (1,))``.

    Raises:
        ValueError: The text after an ``@`` is not a whole number.
    """
    loc: list[str | int] = []
    node_ids: list[str] = []
    item_indexes: list[int] = []
    for part in text.split("/"):
        node_id, *indexes = part.split("@")
        node_ids.append(node_id)
        item_indexes += [int(i) for i in indexes]
        loc += [node_id, *item_indexes[len(item_indexes) - len(indexes):]]
    return tuple(loc), tuple(node_ids), tuple(item_indexes)


def node_path_text(path: Path) -> str:
    """A node path as text, joined by ``/``: ``node_path_text(("loud", "shout"))`` returns ``"loud/shout"``.

    A run keeps each node's outcomes, one per call, under this text, and a saved run writes them under it.
    """
    return "/".join(path)


def place_text(path: Path) -> str:
    """The path of an output or an exclusive group as text, for a person to read: the node's path as text, a dot,
    then the name. ``place_text(("lines", "result"))`` returns ``"lines.result"``, and ``place_text(("loud", "shout",
    "result"))`` returns ``"loud/shout.result"``."""
    return f"{node_path_text(path[:-1])}.{path[-1]}"


def parse_node_path(name: str) -> Path:
    """The node path that ``node_path_text`` wrote as ``name``: ``parse_node_path("loud/shout")`` returns ``("loud", "shout")``."""
    return tuple(name.split("/"))
