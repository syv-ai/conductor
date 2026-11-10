# syv-conductor-nodes

Nodes for [Conductor](https://github.com/syv-ai/conductor) that only compute: text, numbers, yes or no, JSON, regular expressions and a decision that sends a value one of two ways.

```bash
uv add syv-conductor-nodes
```

`conductor_nodes.registry` holds all 29. Add it to yours, or add one module's:

```python
import conductor_nodes
from conductor import NodeRegistry, make_graph

registry = NodeRegistry()
registry.add_subregistry(conductor_nodes.text.registry)

graph = registry.load(make_graph(conductor_nodes.text.split, conductor_nodes.text.uppercase))
print(graph.run(text="a,b").outputs)  # {'text-uppercase.result': ['A', 'B']}
```

| Module | Category | Nodes |
|---|---|---|
| `text` | `text` | `text-uppercase`, `text-lowercase`, `text-trim`, `text-length`, `text-concat`, `text-replace`, `text-contains`, `text-split`, `text-join`, `text-reverse` |
| `math` | `math` | `math-add`, `math-subtract`, `math-multiply`, `math-divide`, `math-modulo`, `math-round`, `math-min`, `math-max`, `math-abs` |
| `logic` | `control`, `logic` | `logic-if-empty`, `logic-if-equals`, `logic-not` |
| `json_ops` | `json` | `json-parse`, `json-stringify`, `json-get` |
| `regex_ops` | `regex` | `regex-match`, `regex-replace`, `regex-extract` |
| `decision` | `control` | `decision` |

To pick by category, filter: `conductor_nodes.registry.filter(lambda d: d.category == "math")`.

`decision` gives its `value` on `if_true` or `if_false` by `when`, and what is wired to the other way doesn't run on that row. Its outputs take the type that arrives on `value`:

```python
from conductor_nodes.decision import decision

graph = conductor_nodes.registry.load(make_graph(conductor_nodes.text.length, decision, edges=[("text-length.result", "decision.value")]))
print([field.type for field in graph.describe().nodes["decision"].outputs])  # ['int', 'int']
```

A JSON value is pydantic's `JsonValue`. The regular-expression nodes run on the `regex` package, with a timeout on every call, so a pattern that would freeze `re` fails its node as `pattern_timeout`.

The documentation, the examples and the changelog live in the repository: https://github.com/syv-ai/conductor
