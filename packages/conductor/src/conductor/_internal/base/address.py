"""Four ways to name a place in a graph or a run, and how each is written as text.

The example graph is the README's, ``lines -> shout -> join``, with ``shout`` replaced by a subgraph: a graph
registered as a node definition. The subgraph ``loud`` is the graph ``strip -> shout``, used as one node::

    lines(text: str) -> Series[str]   # called once; returns ["a", " ", "c"]
    loud                              # wired to lines.result: its inner nodes are called once per line
        strip(line: str) -> str | None
        shout(line: str) -> str
    join(texts: Series[str]) -> str   # wired to loud.shout.result, the output of loud's inner node shout

- A **node path** names a node: its id, after the id of each subgraph node it is inside. The inner node ``shout`` is
  ``("loud", "shout")``, written ``"loud/shout"``; ``lines`` is ``("lines",)``, written ``"lines"``. An output or an
  exclusive group of a node is its node path and its name, side by side: the output ``result`` of ``loud/shout`` is
  written ``"loud/shout.result"``.
- **Item indexes** say which call of a node: one index per loop the node is inside, outermost first. The call of
  ``loud/shout`` for the line ``" "`` has the item indexes ``(1,)``.
- A **loc** names one call: the node path and the item indexes in one tuple. Each index is written after the id
  of the node that is inside its loop in the graph where the loop is. That call is ``("loud", 1, "shout")``,
  written ``"loud@1/shout"``: the loop over ``lines.result`` is in the outer graph, where the node inside it is
  ``loud``, so the index follows ``"loud"``. A host is given a loc when it
  is told about one call, as in ``run.not_called``: ``NotCalled(loc=("loud", 1, "shout"), input="line")``.
- An **address** names one output of a node of a graph record, or one field of the record an output returns:
  ``Address("lines", "result")``, written ``"lines.result"``. A graph record writes where each edge comes from as
  one. Inside the engine, an output deep in a graph is its node's path and its name: compiling names a loop and an
  edge's source so, ``(("loud", "shout"), "result")``.

Because these are written with ``.``, ``/``, ``@`` and ``~`` (``~`` comes before the fingerprint of a waiting
call's id, ``"loud@1/shout~3f2a"``), no node id and no name of an input or output contains one of them (``is_name``).
``Name`` holds every name in a graph record to that rule, and an ``Address`` checks its parts when it is made.
"""

from __future__ import annotations

import re
from typing import Annotated

from pydantic import AfterValidator, ConfigDict, Field
from pydantic.dataclasses import dataclass
from pydantic_core import PydanticCustomError

#: The ids on the way to a node, outermost first: the id of each subgraph node it is inside, then the node's own id.
#: ``("loud", "shout")`` is the node ``shout`` inside the subgraph node ``loud``; ``("lines",)`` is a node of the graph
#: itself. An output of a node is its node path and the output's name, side by side.
NodePath = tuple[str, ...]

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
        raise PydanticCustomError("not_a_field", "'{text}' is not the name of a field: a field's name is not empty and does not start with '.'", {"text": text})
    return text


#: A node id, or the name of an input or output: ``"shout"``, ``"text-split"``. Every name in a graph record is one:
#: a node's id, an input's, an output's and a graph input's or output's. Read from a graph record, anything else is
#: refused (``is_name``).
Name = Annotated[str, AfterValidator(_a_name)]

@dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid"))
class Address:
    """One output of a node of a graph, or one field of the record an output returns: where an edge comes from.

    A graph record writes it as ``{"node": "lines", "output": "result"}``, and a field as ``{"node": "read",
    "output": "result", "field": "email"}``. A subgraph node's outputs are the graph outputs its graph record
    declares, each a name like any other output's. Every part is checked when an address is made, from Python as
    from JSON: ``Address("loud/shout", "result")`` raises, as a node id is a name. As text, for a person to read, it
    is what ``place_text`` writes: ``"lines.result"``, ``"read.result.email"``.

    Attributes:
        node: The node's id.
        output: The output's name.
        field: The field of the record the output returns; ``None`` for the whole output.
    """

    node: Name
    output: Name
    field: Annotated[str | None, AfterValidator(_a_field), Field(exclude_if=lambda field: field is None)] = None

    def __str__(self) -> str:
        return place_text((self.node,), self.output, self.field)


def place_text(node: NodePath, name: str | None = None, field: str | None = None) -> str:
    """A node, or an output, exclusive group or record field of it, as text for a person to read and as a saved
    run writes a node: the node path joined by ``/``, then a dot and the name, then a dot and the field.

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
    """The node path that ``place_text`` wrote as ``text``: ``parse_node_path("loud/shout")`` returns ``("loud", "shout")``."""
    return tuple(text.split("/"))
