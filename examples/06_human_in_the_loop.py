"""Example 6: Human-in-the-loop — pause, checkpoint, and resume.

Shows how a node can pause execution to request human input, how the
engine checkpoints state (JSON-serializable, storable in a database),
and how to resume from the checkpoint with the human's response.
"""

import json
from typing import Annotated

from flowengine import GraphEdge, GraphNode, NodeRegistry, compile
from flowengine.errors import FlowPausedException, HumanInputRequired
from flowengine.execution.engine import execute_sync, resume_sync
from flowengine.widgets import Output, Text, Textarea

registry = NodeRegistry()


# ---------------------------------------------------------------------------
# Normal nodes
# ---------------------------------------------------------------------------


@registry.node("draft", version=1, name="Draft", description="Drafts a message")
def draft(
    topic: Annotated[str, Text(label="Topic")],
) -> Annotated[str, Output(label="Draft")]:
    return f"Dear team,\n\nRe: {topic}\n\nThis is the auto-generated draft about {topic}.\n\nBest regards"


@registry.node("send", version=1, name="Send", description="Sends the message")
def send(
    message: Annotated[str, Textarea(label="Message")],
) -> Annotated[str, Output(label="Status")]:
    print(f"  [SEND] Would send:\n    {message[:60]}...")
    return "sent"


# ---------------------------------------------------------------------------
# Approval node — raises HumanInputRequired to pause the flow
# ---------------------------------------------------------------------------


@registry.node("approve", version=1, name="Approval Gate", description="Requires human approval")
def approve(
    text: Annotated[str, Textarea(label="Content to review")],
) -> Annotated[str, Output(label="Approved content")]:
    # This exception is caught by the engine, NOT propagated as an error.
    # The engine checkpoints state and yields a FlowPausedEvent.
    raise HumanInputRequired(
        prompt=f"Please review and approve this content:\n\n{text[:200]}",
        schema={
            "approved": "bool",
            "edited_text": "str (optional — leave empty to use original)",
        },
    )


# ---------------------------------------------------------------------------
# Build flow: draft -> approve -> send
# ---------------------------------------------------------------------------

nodes = [
    GraphNode("n1", "draft@1", {"topic": "Q3 roadmap"}),
    GraphNode("n2", "approve@1", None),
    GraphNode("n3", "send@1", None),
]
edges = [
    GraphEdge("e1", "n1", "n2", "result", "text"),
    GraphEdge("e2", "n2", "n3", "result", "message"),
]

compiled = compile(nodes=nodes, edges=edges, registry=registry)

# ---------------------------------------------------------------------------
# Execute — the flow will pause at the approval node
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Phase 1: Execute until approval ===\n")

    try:
        results = execute_sync(compiled)
        print("ERROR: Should have paused!")
    except FlowPausedException as e:
        checkpoint = e.checkpoint
        print(f"Flow paused at node: {checkpoint['waiting_node_id']}")
        print(f"Prompt: {checkpoint['prompt']}")
        print(f"Expected response schema: {checkpoint['input_schema']}")
        print()

        # The checkpoint is a plain dict — JSON-serializable, store in DB
        checkpoint_json = json.dumps(checkpoint, indent=2)
        print(f"Checkpoint size: {len(checkpoint_json)} bytes (store this in your DB)")
        print()

    # ---------------------------------------------------------------------------
    # Simulate human approval (in production: wait for user input via UI/API)
    # ---------------------------------------------------------------------------

    print("=== Phase 2: Human approves (simulated) ===\n")

    # The human's response becomes the output of the approval node.
    # It flows downstream to the send node via the edge.
    human_response = "APPROVED: Dear team, Re: Q3 roadmap — looks great, ship it!"

    # Restore from checkpoint (could be hours/days later, different process)
    restored_checkpoint = json.loads(checkpoint_json)

    # Re-compile the graph (host app stores the flow definition)
    # In production you'd load nodes/edges from DB and re-compile
    results = resume_sync(compiled, restored_checkpoint, response=human_response)

    print("\n=== Final results ===\n")
    for node_id, result in results.items():
        print(f"  {node_id}: {result}")
