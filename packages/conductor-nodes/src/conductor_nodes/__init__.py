"""``conductor_nodes``: nodes for text, numbers, yes or no, JSON, patterns and a decision.

``registry`` holds every node; each module's ``registry`` holds its own, so a
host takes what it offers::

    import conductor_nodes
    from conductor import NodeRegistry

    mine = NodeRegistry()
    mine.add_subregistry(conductor_nodes.registry)               # everything
    mine.add_subregistry(conductor_nodes.text.registry)          # the text nodes

Each node has a ``category`` (``control``, ``json``, ``logic``, ``math``,
``regex``, ``text``), so ``conductor_nodes.registry.filter(lambda d: d.category
== "math")`` picks by it.
"""

from conductor import NodeRegistry

from conductor_nodes import decision, json_ops, logic, math, regex_ops, text

registry = NodeRegistry()
for _module in (text, math, logic, json_ops, regex_ops, decision):
    registry.add_subregistry(_module.registry)

__all__ = ["decision", "json_ops", "logic", "math", "regex_ops", "registry", "text"]
