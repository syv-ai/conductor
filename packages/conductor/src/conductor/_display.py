"""How a node and a registry show themselves, in a terminal and in a notebook.

The rule is pydantic's and scikit-learn's: an object prints as the call that
states what it is, leaving out what is at its default, and a container prints
its children's own reprs. A record gets that from ``ConductorModel``; a node
is a class, so its text comes from ``node_repr`` through the node metaclass,
and a registry composes the reprs of its nodes. In a notebook both render
``nodes_table``: one row per node, with the current version's inputs and
outputs.
"""

from __future__ import annotations

from collections.abc import Iterable
from html import escape
from typing import Any


def node_repr(cls: Any) -> str:
    """``Greet(id='greet', title='Greeting', category='text', versions=(1,))``.

    ``tags`` and ``deprecation`` appear only when the class has them.
    """
    parts = [f"id={cls.id!r}", f"title={cls.title!r}", f"category={cls.category!r}"]
    if cls.tags:
        parts.append(f"tags={tuple(cls.tags)!r}")
    parts.append(f"versions={tuple(sorted(cls.versions))!r}")
    if cls.deprecation is not None:
        parts.append(f"deprecation={cls.deprecation!r}")
    return f"{cls.__name__}({', '.join(parts)})"


def nodes_table(nodes: Iterable[Any]) -> str:
    """An HTML table of ``nodes``: id, title, category, versions, and the current version's inputs → outputs."""
    header = "".join(f"<th>{name}</th>" for name in ("id", "title", "category", "versions", "inputs → outputs"))
    rows = "".join(_row(cls) for cls in nodes)
    return f"<table><tr>{header}</tr>{rows}</table>"


def _row(cls: Any) -> str:
    interface = cls.versions[cls.current].interface
    signature = f"{', '.join(i.name for i in interface.inputs)} → {', '.join(o.name for o in interface.outputs)}"
    cells = (cls.id, cls.title, cls.category, ", ".join(str(n) for n in sorted(cls.versions)), signature)
    return "<tr>" + "".join(f"<td>{escape(str(cell))}</td>" for cell in cells) + "</tr>"
