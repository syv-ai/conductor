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
- An **address** names one input or output of a node: ``"lines.result"``, ``"shout.line"``. A graph record
  writes its edges and its constants with addresses.

Because these are written with ``.``, ``/``, ``@`` and ``~`` (``~`` comes before the fingerprint of a waiting
call's id, ``"loud@1/shout~3f2a"``), no node id and no name of an input or output holds one of them (``is_name``).
"""

from __future__ import annotations

import re

#: The ids that name one node: the id of each subgraph node it is inside, outermost first, and its own id last.
#: ``("loud", "shout")`` is the node ``shout`` inside the subgraph node ``loud``; ``("lines",)`` is a node of the
#: graph itself.
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
NAME_PATTERN = r"^[^./@~]+$"

_NAME = re.compile(NAME_PATTERN)


def is_name(text: str) -> bool:
    """Whether ``text`` can be a node id or the name of an input or output (``NAME_PATTERN``)."""
    return _NAME.match(text) is not None


def split_address(address: str) -> tuple[str, str]:
    """Split an address at its first dot into the node id and the name after it.

    ``split_address("shout.line")`` returns ``("shout", "line")``. A node id has no dot in it, so everything after
    the first dot is the name, which may have dots of its own: ``split_address("read.result.name")`` returns
    ``("read", "result.name")``, the field ``name`` of the record the output ``result`` gives, and
    ``split_address("loud.shout.result")`` returns ``("loud", "shout.result")``, the output of the inner node
    ``shout`` of the subgraph node ``loud``. Text with no dot is a node id alone:
    ``split_address("shout")`` returns ``("shout", "")``.
    """
    node_id, _, name = address.partition(".")
    return node_id, name


def loc_text(loc: Loc) -> str:
    """A loc as text: ``/`` between node ids and ``@`` before each index.

    ``loc_text(("loud", 1, "shout"))`` returns ``"loud@1/shout"``, and ``loc_text(("shout", 0, 2))`` returns
    ``"shout@0@2"``. A loc starts with a node id.
    """
    return "".join(f"@{part}" if isinstance(part, int) else f"/{part}" for part in loc)[1:]


def parse_loc_text(text: str) -> tuple[Loc, NodePath, ItemIndexes]:
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


def node_path_text(path: NodePath) -> str:
    """A node path as text, joined by ``/``: ``node_path_text(("loud", "shout"))`` returns ``"loud/shout"``.

    A run keeps each node's outcomes, one per call, under this text, and a saved run writes them under it.
    """
    return "/".join(path)


def parse_node_path(name: str) -> NodePath:
    """The node path that ``node_path_text`` wrote as ``name``: ``parse_node_path("loud/shout")`` returns ``("loud", "shout")``."""
    return tuple(name.split("/"))
