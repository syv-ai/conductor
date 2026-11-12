"""A graph as a host stores it: ``graph.py`` is ``GraphRecord``, ``node.py`` is ``NodeRecord``.

A graph record is its nodes and its edges, and nothing else::

    {
        "nodes": [
            {"id": "lines", "node": "lines"},
            {"id": "shout", "node": "shout", "constants": {"mark": "?"}},
            {"id": "join", "node": "join"}
        ],
        "edges": [["lines.result", "shout.line"], ["shout.result", "join.texts"]]
    }

A node names the node definition it uses (``node``) and its version (``version``, 1 when left out). A constant
(``constants``) is the value an input has on every call in every run: no caller can change it. A default
(``defaults``) is the value of a graph input when the caller gives none. ``display`` is the host's own: conductor
stores it and never reads it. An empty field is left out when the record is saved, so a bare node is
``{"id": "join", "node": "join"}``.

Edges are a list of ``[source, target]`` addresses, ``node.input`` or ``node.output``, or ``node.output.field`` for
one field of a record an output gives. They are kept sorted, so two records that differ only in the order of their
edges are equal and save alike. An edge listed twice stays twice, and compiling reports it (``repeated_edge``).
Several edges into one input are alternatives, from opposite branches of one decision, so nothing depends on the
order of the list.

A host hands a record in as a ``GraphRecord`` or its JSON text; ``gate/record.py`` reads it. ``Graph.compile``
compiles it into a ``Graph``, and what the record says that cannot be meant (a repeated edge, an edge naming a node
the graph does not have, an input given both a constant and a default) is a problem of that graph.
"""

from .graph import GraphRecord
from .node import NodeRecord
