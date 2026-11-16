"""The graph record: a graph written down as plain data, the way a host saves it.

A graph is a set of nodes wired together. A host, such as an editor, keeps that drawing as a graph record. It is a
small, frozen pydantic model that turns into JSON and back. Here is a graph that splits a text into lines, shouts
each line and joins them again::

    {
        "nodes": {
            "lines": {"node": "lines", "inputs": {"text": {"graph_input": "text"}}},
            "shout": {"node": "shout", "inputs": {
                "line": {"sources": [{"node": "lines", "output": "result"}]},
                "mark": {"constant": "?"}
            }},
            "join": {"node": "join", "inputs": {"texts": {"sources": [{"node": "shout", "output": "result"}]}}}
        },
        "outputs": {"result": {"node": "join", "output": "result"}}
    }

Each node names the node it uses, and says what each of its inputs gets. An input gets a constant, a graph input
the caller gives, or the output of another node through an edge. ``GraphRecord`` (``graph.py``) is the whole
record, and ``NodeRecord`` and ``InputEntry`` (``node.py``) are one node and one input.

A record only says what is written down. It doesn't know what the nodes are. ``validator.py`` checks that a record
has the right shape, for example that no node id is written twice. Whether it makes sense as a graph, for example
that an edge comes from a node that exists, is checked later, when ``Graph.compile`` compiles it. The full rules
are in the reference: ``python -m conductor.about the-graph-record``.
"""

from .graph import GraphRecord
from .node import InputEntry, NodeRecord
from .validator import RecordValidator, WrittenValuesValidator, read_json

__all__ = [
    "GraphRecord",
    "InputEntry",
    "NodeRecord",
    "RecordValidator",
    "WrittenValuesValidator",
    "read_json",
]
