"""conductor-providers: adapters that sit between conductor and the outside world.

``conductor_providers.mermaid.flowchart(compiled)`` draws a compiled graph
as a Mermaid flowchart, for a person reading it in a notebook, a pull
request or a bug report.
"""

from conductor_providers import mermaid

__all__ = ["mermaid"]
