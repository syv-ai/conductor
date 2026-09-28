<p align="center">
  <img src="assets/logo-white-background.png" alt="Conductor logo" width="140">
</p>

# Conductor

A host-agnostic Python engine that compiles and runs graphs of typed nodes. A node is a class whose typed `run` signature is its interface. You place nodes in a graph, compile it into a record that says everything about it, and run it as a stream of events, once per row wherever a series arrives.

```python
from typing import Annotated

from conductor import CompiledGraph, Graph, GraphNode, NodeDefinition, NodeRegistry, Param, Result, run_sync, Static
from conductor.widgets import TextWidget
from conductor_nodes.types import Number, Text


class Length(NodeDefinition):
    id = "length"
    title = "Length"
    description = "Counts characters"
    category = "text"

    def run(self, text: Annotated[Text, Param(title="Text", widget=TextWidget())]) -> Annotated[Number, Result(title="Length")]:
        return Number(len(text))


registry = NodeRegistry()
registry.register(Length)

compiled = CompiledGraph.from_graph(
    Graph(nodes=[GraphNode(id="size", type="length", version=1, bindings={"text": Static(["a", "bcd"])})]),
    registry,
)
assert compiled.is_runnable, compiled.problems
list(run_sync(compiled).state.results(compiled)["size"]["result"])   # [1.0, 3.0]: one run per value typed in
```

## Read next

- [Overview](OVERVIEW.md): the architecture on one page.
- [Widgets](widgets.md): the controls, and how an input declares one.
- `python -m conductor.about`: the library reference that ships in the wheel.
- The notebooks in [`examples/`](https://github.com/syv-ai/conductor/tree/main/examples) cover nodes, graphs, versions, a person in the loop and widgets.
