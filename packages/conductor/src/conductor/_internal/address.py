"""Three ways to say where something is in a run, and the text form of each.

The example used in this file: a graph has a node ``e``, which is a subgraph (a stored graph used as
one node). ``e`` is called once per document. Inside it, a node ``ask`` is called once per line of the document.

- A **node path** names a node: ``("e", "ask")``. As text it is ``"e/ask"``.
- **Item indexes** say which call of that node: ``(0, 2)`` is the call for document 0, line 2.
- A **loc** is both in one tuple: ``("e", 0, "ask", 2)``. After each node id come the positions that the
  node adds to the item indexes: ``e`` is called per document, so one position follows it, and ``ask`` adds one
  more for the line. As text it is ``"e@0/ask@2"``. A loc is what the application that uses the library is
  given when it is told about one call, for example a call that waits for a person's answer.
"""

from __future__ import annotations

#: The ids that name one node: its own id last, and before it the id of each subgraph it is inside,
#: outermost first. ``("e", "ask")`` is the node ``ask`` inside the subgraph ``e``; ``("load",)`` is a
#: node of the graph itself.
NodePath = tuple[str, ...]

#: Which call of a node: the position of its item in each loop the node is inside, outermost first.
#: ``(0, 2)`` is item 2 of the inner list, for item 0 of the outer list. A node inside no loop has ``()``.
ItemIndexes = tuple[int, ...]

#: A node path and a set of item indexes in one tuple: each node id, followed by the positions that node adds to the
#: item indexes. ``("e", 0, "ask", 2)`` is the node path ``("e", "ask")`` with the item indexes ``(0, 2)``. A node
#: inside two loops of its own graph has two positions after its id: ``("clean", 0, 1)``.
Loc = tuple[str | int, ...]


def loc_text(loc: Loc) -> str:
    """A loc as text: ``@`` before each position and ``/`` between nodes.

    ``loc_text(("e", 0, "ask", 2))`` returns ``"e@0/ask@2"``, and ``loc_text(("clean", 0, 1))`` returns
    ``"clean@0@1"``.
    """
    return "".join(f"@{step}" if isinstance(step, int) else f"/{step}" for step in loc)[1:]


def parse_loc_text(text: str) -> tuple[Loc, NodePath, ItemIndexes]:
    """Read the text of a loc back, and also split it into its node path and its item indexes.

    ``parse_loc_text("e@0/ask@2")`` returns ``(("e", 0, "ask", 2), ("e", "ask"), (0, 2))``.

    Raises:
        ValueError: The text after an ``@`` is not a number.
    """
    loc: list[str | int] = []
    node_ids: list[str] = []
    item_indexes: list[int] = []
    for step in text.split("/"):
        node_id, *positions = step.split("@")
        node_ids.append(node_id)
        item_indexes += [int(i) for i in positions]
        loc += [node_id, *item_indexes[len(item_indexes) - len(positions):]]
    return tuple(loc), tuple(node_ids), tuple(item_indexes)


def node_path_text(path: NodePath) -> str:
    """A node path as text, joined by ``/``: ``node_path_text(("e", "ask"))`` returns ``"e/ask"``.

    Dicts that hold something per node, in a run and in a saved run, have this text as their key.
    """
    return "/".join(path)


def parse_node_path(name: str) -> NodePath:
    """The node path that ``node_path_text`` wrote as ``name``: ``parse_node_path("e/ask")`` returns ``("e", "ask")``."""
    return tuple(name.split("/"))
