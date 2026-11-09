<p align="center">
  <img src="assets/logo-white-background.png" alt="Conductor logo" width="140">
</p>

# Conductor

Compile and run graphs of typed Python functions. A node is a function and its type hints are its interface. You put nodes in a graph, compile it into a record that says what's wrong and where everything runs, and run it as a stream of events, once per row wherever a column arrives.

```python
from conductor import CompiledGraph, Graph, NodeRegistry, node


@node("length")
def length(text: str) -> int:
    return len(text)


compiled = CompiledGraph.from_graph(Graph(nodes=[length.graph_node("size", text="abcd")]), NodeRegistry([length]))
print(compiled.outputs(compiled.run_sync()))  # {'size.result': 4}
```

## Read next

- [Overview](OVERVIEW.md): the architecture on one page.
- [Widgets](widgets.md): the controls, and how an input declares one.
- [Serving runs over HTTP](serving-runs.md): a page of routes to copy.
- `python -m conductor.about`: the library reference that ships in the wheel.
- The notebooks in [`examples/`](https://github.com/syv-ai/conductor/tree/main/examples) cover nodes, graphs, tables, versions, a person in the loop and widgets.
