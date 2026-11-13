"""A graph as a host stores it: ``graph.py`` is ``GraphRecord``, ``node.py`` is ``NodeRecord`` and what each input gets.

A graph record is its nodes by id and its outputs by name::

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

A node names the node definition it uses (``node``) and its version (``version``, 1 when left out). Each input it
lists gets an ``InputEntry``, exactly one of three:

- a constant, the value the input has on every call in every run: no caller can change it;
- a graph input, the value the caller gives when a run starts, under the graph input's name, with a default for
  when the caller gives none;
- the sources of the edges into it, each a ``Source``: the output it comes from, or one field of the record an
  output returns: ``{"node": "read", "output": "result", "field": "email"}``.

An input the record leaves out gets the node's own default. ``display`` is the host's own: conductor stores it and
never reads it. An empty field of a node is left out when the record is saved, so a bare node is ``{"node":
"join"}``. ``outputs`` names the graph's outputs, each a node's output; used as a subgraph, the graph's inputs and
outputs are the node's.

A host hands a record in as a ``GraphRecord`` or its JSON text; ``gate/record.py`` reads it, and refuses a record
that is not this shape: a key written twice in one object (two nodes with one id), a node id or input that is not a
name, a source listed twice, or an input given more than one of the three. ``Graph.compile`` compiles a record it
read, and what the record says that cannot be meant as a graph (an edge from a node the graph does not have, an
input the node does not have, a graph input given two types) is a problem of that graph.
"""

from .graph import GraphRecord
from .node import InputEntry, NodeRecord

__all__ = ["GraphRecord", "InputEntry", "NodeRecord"]
