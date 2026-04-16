"""Example 5: Auto-discovery and JSON schema for frontends.

Shows how to use registry.discover() to auto-register nodes from a
package, and how to serialize the registry to JSON for a frontend.
"""

from typing import Annotated

from flowengine import NodeRegistry
from flowengine.registry.schema import serialize_registry
from flowengine.widgets import Output, Text

# ---------------------------------------------------------------------------
# Option A: Manual registration (what you've seen in other examples)
# ---------------------------------------------------------------------------

registry = NodeRegistry()


@registry.node("echo", version=1, name="Echo", description="Echoes input")
def echo(text: Annotated[str, Text(label="Input")]) -> Annotated[str, Output(label="Output")]:
    return text


@registry.node("echo", version=2, name="Echo v2", description="Echo with prefix support")
def echo_v2(
    text: Annotated[str, Text(label="Input")],
    prefix: Annotated[str, Text(label="Prefix")] = "",
) -> Annotated[str, Output(label="Output")]:
    return f"{prefix}{text}"


# ---------------------------------------------------------------------------
# Option B: Auto-discovery
#
# If you have a package structure like:
#
#   myapp/
#     nodes/
#       __init__.py
#       text.py      # contains @registry.node("echo", ...) etc.
#       math.py      # contains @registry.node("add", ...) etc.
#
# You can register all nodes at once:
#
#   registry.discover("myapp.nodes")
#
# This imports every module in the package, triggering all decorators.
# ---------------------------------------------------------------------------

# (We can't demonstrate this without a real package, but that's the API)


# ---------------------------------------------------------------------------
# JSON schema for frontends
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Serialize the entire registry to JSON
    schema = serialize_registry(registry)

    print("=== Registry JSON (for frontend) ===\n")
    print(json.dumps(schema, indent=2))

    # The frontend uses this to:
    # 1. Build the node palette (name, description, tags)
    # 2. Render input widgets (widget type, label, choices, etc.)
    # 3. Render output handles (name, label)
    # 4. Show deprecation warnings (deprecated flag)

    print(f"\n=== Summary ===")
    print(f"Total node versions: {len(schema)}")
    for node in schema:
        status = "DEPRECATED" if node["deprecated"] else "current"
        print(f"  {node['id']}: {node['name']} ({status})")
        for inp in node["inputs"]:
            opt = " (optional)" if inp["optional"] else ""
            print(f"    in:  {inp['name']}: {inp['type']} [{inp['widget']}]{opt}")
        for out in node["outputs"]:
            print(f"    out: {out['name']}: {out['type']}")
