"""The graph record: a graph written down as plain data, the way a host saves it.

A graph is a set of nodes wired together. A host, such as an editor, keeps that drawing as a graph record. It is a
small, frozen pydantic model that turns into JSON and back. Here is a graph that splits a text into lines, shouts
each line and joins them again::

    {
        "inputs": {"text": {}},
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

The record declares the graph's inputs and outputs. Each node names the node it uses, and says what each of its
inputs gets: a constant, a graph input the caller gives, or the output of another node through an edge.
``GraphRecord`` and ``GraphInputRecord`` (``graph.py``) are the whole record and one graph input. ``NodeRecord`` and
``InputEntry`` (``node.py``) are one node and one input, and an entry is a ``ConstantEntry``, a ``GraphInputEntry``
or a ``SourcesEntry``.

A record only says what is written down. It doesn't know what the nodes are. ``validator.py`` checks that a record
has the right shape, for example that no node id is written twice. Whether it makes sense as a graph, for example
that an entry names a graph input the record declares, is checked later, when ``Graph.compile`` compiles it. The
full rules are in the reference: ``python -m conductor.about the-graph-record``.
"""

from .graph import GraphInputRecord, GraphRecord
from .node import ConstantEntry, GraphInputEntry, InputEntry, NodeRecord, SourcesEntry
from .validator import RecordValidator, WrittenValuesValidator, read_json, refused_record

__all__ = [
    "ConstantEntry",
    "GraphInputEntry",
    "GraphInputRecord",
    "GraphRecord",
    "InputEntry",
    "NodeRecord",
    "RecordValidator",
    "SourcesEntry",
    "WrittenValuesValidator",
    "read_json",
    "refused_record",
]
