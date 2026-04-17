"""Example 3: Class-based nodes and FlowStore.

Shows how to create nodes using the BaseNode ABC (for complex nodes
that need internal state or access to FlowRunState), and how to use
the FlowStore for cross-node data sharing outside of edges.
"""

from typing import Annotated, Any

from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor.execution.engine import execute_sync
from conductor.execution.request import NodeExecRequest
from conductor.execution.store import FlowStore
from conductor.node import BaseNode
from conductor.widgets import Output, Text

registry = NodeRegistry()


# ---------------------------------------------------------------------------
# Class-based node: has access to the full NodeExecRequest
# ---------------------------------------------------------------------------


class WordCounter(BaseNode):
    """Counts words in input text and stores the count in FlowStore."""

    node_id = "word-counter"
    node_name = "Word Counter"
    node_description = "Counts words and caches the count"

    def execute(self, req: NodeExecRequest) -> Any:
        text = req.inputs.get("text", "")
        count = len(text.split())

        # Store the count in FlowStore for other nodes to access
        req.state.store.set(f"word_count:{req.node_id}", count)

        return count


registry.register_class(WordCounter)


# ---------------------------------------------------------------------------
# Function node using FlowStore via type injection
# ---------------------------------------------------------------------------


@registry.node("echo", version=1, name="Echo", description="Echoes input")
def echo(text: Annotated[str, Text(label="Input")]) -> Annotated[str, Output(label="Output")]:
    return text


@registry.node(
    "summary-report",
    version=1,
    name="Summary Report",
    description="Generates a report using data from FlowStore",
)
def summary_report(
    text: Annotated[str, Text(label="Text")],
    store: FlowStore,  # Auto-injected — NOT a node input, won't appear in UI
) -> Annotated[str, Output(label="Report")]:
    # Read data that was stored by upstream nodes
    all_keys = store.keys()
    word_counts = {k: store.get(k) for k in all_keys if k.startswith("word_count:")}

    report_lines = [f"Text: {text[:50]}..."]
    for key, count in word_counts.items():
        node_id = key.split(":", 1)[1]
        report_lines.append(f"  Node {node_id} counted {count} words")

    return "\n".join(report_lines)


# ---------------------------------------------------------------------------
# Build a flow:
#   echo("some long text...") -> word-counter -> summary-report
#
# The word-counter stores data in FlowStore.
# The summary-report reads it back via store injection.
# ---------------------------------------------------------------------------

long_text = "The quick brown fox jumps over the lazy dog and then does it again"

nodes = [
    GraphNode("n1", "echo@1", {"text": long_text}),
    GraphNode("n2", "word-counter@1", None),
    GraphNode("n3", "summary-report@1", None),
]

edges = [
    GraphEdge("e1", "n1", "n2", "result", "text"),
    GraphEdge("e2", "n1", "n3", "result", "text"),  # text goes to report too
]

if __name__ == "__main__":
    compiled = compile(nodes=nodes, edges=edges, registry=registry)
    results = execute_sync(compiled)

    print("=== Results ===")
    for node_id, result in results.items():
        print(f"\n{node_id}:")
        print(f"  {result}")

    # Verify that FlowStore is NOT visible as a node input
    report_def = registry.get("summary-report@1")
    input_names = [inp.name for inp in report_def.inputs]
    print(f"\nSummary report inputs: {input_names}")
    assert "store" not in input_names, "FlowStore should not appear as a node input"
    print("(FlowStore correctly excluded from inputs)")
