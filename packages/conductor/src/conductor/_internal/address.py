"""Three ways to say where something is in a run, and the text form of each.

The example used in this file: a graph has a node ``e``, which is an embedded graph (a stored graph used as
one node). ``e`` is called once per document. Inside it, a node ``ask`` is called once per line of the document.

- A **node path** names a node: ``("e", "ask")``. As text it is ``"e/ask"``.
- An **item path** says which call of that node: ``(0, 2)`` is the call for document 0, line 2.
- A **loc** is both in one tuple, each node id followed by the positions in that node's own loops:
  ``("e", 0, "ask", 2)``. As text it is ``"e@0/ask@2"``. This is the form a host sees, for example in
  ``run.not_called`` and in the id of a question that waits for an answer.
"""

from __future__ import annotations

#: The ids that name one node: its own id last, and before it the id of each embedded graph it is inside,
#: outermost first. ``("e", "ask")`` is the node ``ask`` inside the embedded graph ``e``; ``("load",)`` is a
#: node of the graph itself.
NodePath = tuple[str, ...]

#: Which call of a node: the position of its item in each loop the node is inside, outermost first.
#: ``(0, 2)`` is item 2 of the inner list, for item 0 of the outer list. A node inside no loop has ``()``.
ItemPath = tuple[int, ...]

#: A node path and an item path in one tuple: each node id, followed by the positions in that node's own
#: loops. ``("e", 0, "ask", 2)`` is the node path ``("e", "ask")`` with the item path ``(0, 2)``.
Loc = tuple[str | int, ...]


def loc_text(loc: Loc) -> str:
    """A loc as text: ``@`` before each position and ``/`` between nodes.

    ``loc_text(("e", 0, "ask", 2))`` returns ``"e@0/ask@2"``.
    """
    return "".join(f"@{step}" if isinstance(step, int) else f"/{step}" for step in loc)[1:]


def parse_loc_text(text: str) -> tuple[Loc, NodePath, ItemPath]:
    """Read the text of a loc back, and also split it into its node path and its item path.

    ``parse_loc_text("e@0/ask@2")`` returns ``(("e", 0, "ask", 2), ("e", "ask"), (0, 2))``.

    Raises:
        ValueError: The text after an ``@`` is not a number.
    """
    loc: list[str | int] = []
    node_ids: list[str] = []
    item_path: list[int] = []
    for step in text.split("/"):
        node_id, *positions = step.split("@")
        node_ids.append(node_id)
        item_path += [int(i) for i in positions]
        loc += [node_id, *item_path[len(item_path) - len(positions):]]
    return tuple(loc), tuple(node_ids), tuple(item_path)


def node_name(path: NodePath) -> str:
    """A node path as text, joined by ``/``: ``node_name(("e", "ask"))`` returns ``"e/ask"``.

    A run and a saved run use this text as the node's key.
    """
    return "/".join(path)


def node_path(name: str) -> NodePath:
    """The node path that ``node_name`` wrote as ``name``: ``node_path("e/ask")`` returns ``("e", "ask")``."""
    return tuple(name.split("/"))
