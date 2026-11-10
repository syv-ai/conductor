"""``conductor_nodes``: nodes for text, numbers, yes or no, JSON, patterns and a decision.

``registry`` holds every node; each module's ``registry`` holds its own, so a
host takes what it offers::

    import conductor_nodes
    from conductor import NodeRegistry

    mine = NodeRegistry()
    mine.add_subregistry(conductor_nodes.registry)               # everything
    mine.add_subregistry(conductor_nodes.text.registry)          # the text nodes

Each module holds one ``category`` of node (``control``, ``json``, ``logic``,
``math``, ``regex``, ``text``), so a module's ``registry`` and
``conductor_nodes.registry.filter(lambda d: d.category == "math")`` pick the
same nodes.
"""

from conductor import NodeRegistry

from conductor_nodes import control, json_ops, logic, math, regex_ops, text

registry = NodeRegistry()
for _module in (text, math, logic, json_ops, regex_ops, control):
    registry.add_subregistry(_module.registry)

__all__ = ["control", "json_ops", "logic", "math", "regex_ops", "registry", "text"]
