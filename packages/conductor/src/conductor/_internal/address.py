"""Where something is in a graph and in a run of it: the tuples the engine holds, and each as the text a host and a save hold."""

from __future__ import annotations

#: A node whatever item it runs for: its id, after the id of each embedded graph's node it is inside, outermost
#: first, ``("e", "ask")``.
NodePath = tuple[str, ...]

#: The place of an item in each map a node runs in, outermost first; ``()`` for a node that runs once.
ItemPath = tuple[int, ...]

#: Where something is in a run: a node's id, then the place of its item in each of its scope's maps; for what is
#: inside an embedded graph, the embedding node and its item first, ``("e", 0, "ask", 2)``.
Loc = tuple[str | int, ...]


def loc_text(loc: Loc) -> str:
    """A loc as a waiting id writes it: each node id, ``@`` before each place of its item, and ``/`` between
    nodes, ``e@0/ask@2``."""
    return "".join(f"@{step}" if isinstance(step, int) else f"/{step}" for step in loc)[1:]


def parse_loc_text(place: str) -> tuple[Loc, NodePath, ItemPath]:
    """The place a waiting id names, ``e@0/ask@2``: as a loc, then the ids of its nodes and the places of its
    items apart, which name the node and its instance.

    Raises:
        ValueError: A place in it isn't a number.
    """
    loc: list[str | int] = []
    nodes: list[str] = []
    key: list[int] = []
    for step in place.split("/"):
        node_id, *places = step.split("@")
        nodes.append(node_id)
        key += [int(i) for i in places]
        loc += [node_id, *key[len(key) - len(places):]]
    return tuple(loc), tuple(nodes), tuple(key)


def node_name(path: NodePath) -> str:
    """A node as a run and a save name it: its id after the ids of the embedded graphs' nodes it is inside,
    joined by ``/``, ``e/ask``."""
    return "/".join(path)


def node_path(name: str) -> NodePath:
    """The path of the node a run and a save name ``name``."""
    return tuple(name.split("/"))
