<p align="center">
  <img src="assets/logo-white-background.png" alt="Conductor logo" width="140">
</p>

# Conductor

Run graphs of typed Python functions. A node is a function and its type hints are its interface. You write a graph as data, its nodes and edges, load it into a graph that says what's wrong and where everything runs, and run it, once per row wherever a column arrives.

```python
from conductor import NodeRegistry, make_graph

registry = NodeRegistry()


@registry.node("length")
def length(text: str) -> int:
    return len(text)


graph = registry.load(make_graph(length, values={"length.text": "abcd"}))
print(graph.run().outputs)  # {'length.result': 4}
```

## Read next

- [Overview](OVERVIEW.md): the architecture on one page.
- [Widgets](widgets.md): the controls, and how an input declares one.
- [Serving runs over HTTP](serving-runs.md): a page of routes to copy.
- `python -m conductor.about`: the library reference that ships in the wheel.
- The notebooks in [`examples/`](https://github.com/syv-ai/conductor/tree/main/examples) cover nodes, graphs, tables, versions, a person in the loop and widgets.
