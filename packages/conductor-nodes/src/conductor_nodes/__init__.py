"""Ready-made nodes for conductor: text, numbers, yes or no, JSON, patterns, loops, tables and branches.

None of them call out to anything. They only work on the values they are given.

``registry`` has every node. Each module also has a ``registry`` of its own, so you can add all the nodes or
just some::

    import conductor_nodes
    from conductor import NodeRegistry

    mine = NodeRegistry()
    mine.add_subregistry(conductor_nodes.registry)               # everything
    mine.add_subregistry(conductor_nodes.text.registry)          # only the text nodes

Each module is one ``category``: ``control``, ``json``, ``logic``, ``math``, ``regex``, ``series``, ``table`` and
``text``. So ``conductor_nodes.registry.filter(lambda d: d.category == "math")`` gives the same nodes as
``conductor_nodes.math.registry``.
"""

from conductor import NodeRegistry

from conductor_nodes import control, json_ops, logic, math, regex_ops, series, tables, text

registry = NodeRegistry()
for _module in (text, math, logic, json_ops, regex_ops, series, tables, control):
    registry.add_subregistry(_module.registry)

__all__ = [
    "control",
    "json_ops",
    "logic",
    "math",
    "regex_ops",
    "registry",
    "series",
    "tables",
    "text",
]
