"""``make_graph``: a graph record written from Python, its graph inputs and outputs declared for it.

It stands above every other package of the core, as a convenience on top of it, and nothing in the core imports it.

``make.py`` has ``make_graph``, and ``validator.py`` ``ArgumentsValidator``, the check of its arguments.
"""

from .make import make_graph

__all__ = ["make_graph"]
