"""Example 4: Control flow — conditionals and for-each loops.

Shows how to use the SKIPPED sentinel for conditional branching
and CompoundNodeType for for-each loop iteration.
"""

from typing import Annotated

from flowengine import GraphEdge, GraphNode, NodeRegistry, compile
from flowengine._sentinel import SKIPPED
from flowengine.compound.for_each import FOR_EACH
from flowengine.execution.engine import execute_sync
from flowengine.types import NodeCategory
from flowengine.widgets import ConnectionList, Dropdown, Output, Text

registry = NodeRegistry()


# ---------------------------------------------------------------------------
# IO nodes
# ---------------------------------------------------------------------------


@registry.node("echo", version=1, name="Echo", description="Echoes input")
def echo(text: Annotated[str, Text(label="Input")]) -> Annotated[str, Output(label="Output")]:
    return text


@registry.node("upper", version=1, name="Upper", description="Uppercases")
def upper(text: Annotated[str, Text(label="Input")]) -> Annotated[str, Output(label="Output")]:
    return text.upper()


@registry.node("prefix", version=1, name="Prefix", description="Adds prefix")
def prefix(text: Annotated[str, Text(label="Input")]) -> Annotated[str, Output(label="Output")]:
    return f">> {text}"


# ---------------------------------------------------------------------------
# Conditional node: outputs SKIPPED on the inactive branch
# ---------------------------------------------------------------------------


@registry.node(
    "if-empty",
    version=1,
    name="If Empty",
    description="Routes based on whether input is empty",
    category=NodeCategory.CONTROL,
)
def if_empty(
    text: Annotated[str, Text(label="Input")],
) -> tuple[
    Annotated[str, Output(label="Not empty")],
    Annotated[str, Output(label="Empty")],
]:
    if text.strip():
        return (text, SKIPPED)      # "Not empty" branch active
    else:
        return (SKIPPED, "empty")   # "Empty" branch active


# ---------------------------------------------------------------------------
# For-each loop markers (the engine handles iteration via compound nodes)
# ---------------------------------------------------------------------------


@registry.node(
    "for-each-start",
    version=1,
    name="For Each (Start)",
    description="Iterates over items",
    category=NodeCategory.CONTROL,
)
def for_each_start(
    items: Annotated[list[str], ConnectionList(label="Items")],
) -> tuple[
    Annotated[str, Output(label="Item")],
    Annotated[int, Output(label="Index")],
]:
    raise NotImplementedError("Handled by compound node")


@registry.node(
    "for-each-end",
    version=1,
    name="For Each (End)",
    description="Collects results",
    category=NodeCategory.CONTROL,
)
def for_each_end(
    item: Annotated[str, Text(label="Item")],
) -> Annotated[list[str], Output(label="Collected")]:
    raise NotImplementedError("Handled by compound node")


# ===================================================================
# Example A: Conditional branching
# ===================================================================
#
#   echo("hello") -> if-empty -> (not empty) -> upper
#                             -> (empty)     -> prefix
#
# Since "hello" is not empty, upper runs and prefix is SKIPPED.

print("=== Conditional branching ===")

nodes_a = [
    GraphNode("text", "echo@1", {"text": "hello"}),
    GraphNode("cond", "if-empty@1", None),
    GraphNode("up", "upper@1", None),
    GraphNode("pf", "prefix@1", None),
]
edges_a = [
    GraphEdge("e1", "text", "cond", "result", "text"),
    GraphEdge("e2", "cond", "up", "output_1", "text"),    # "not empty" branch
    GraphEdge("e3", "cond", "pf", "output_2", "text"),    # "empty" branch
]

compiled_a = compile(nodes=nodes_a, edges=edges_a, registry=registry)
results_a = execute_sync(compiled_a)

for nid, res in results_a.items():
    print(f"  {nid}: {res}")

# up should have "HELLO", pf should not appear (skipped)
print()


# ===================================================================
# Example B: For-each loop
# ===================================================================
#
#   for-each-start(["alice", "bob", "charlie"])
#       -> upper (loop body)
#       -> for-each-end
#
# Result: ["ALICE", "BOB", "CHARLIE"]

print("=== For-each loop ===")

nodes_b = [
    GraphNode("start", "for-each-start@1", {"items": ["alice", "bob", "charlie"]}),
    GraphNode("body", "upper@1", None),
    GraphNode("end", "for-each-end@1", None),
]
edges_b = [
    GraphEdge("e1", "start", "body", "output_1", "text"),
    GraphEdge("e2", "body", "end", "result", "item"),
]

compiled_b = compile(
    nodes=nodes_b,
    edges=edges_b,
    registry=registry,
    compound_types=[FOR_EACH],  # Register the for-each compound handler
)
results_b = execute_sync(compiled_b)

for nid, res in results_b.items():
    print(f"  {nid}: {res}")
