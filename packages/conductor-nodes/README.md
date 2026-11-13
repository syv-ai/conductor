# syv-conductor-nodes

Nodes for [Conductor](https://github.com/syv-ai/conductor) that only compute: text, numbers, yes or no, JSON, regular expressions, tables and a decision that sends a value one of two ways.

```bash
uv add syv-conductor-nodes
```

`conductor_nodes.registry` has all 34. Add it to yours, or add one module's:

```python
import conductor_nodes
from conductor import Graph, NodeRegistry, make_graph

registry = NodeRegistry()
registry.add_subregistry(conductor_nodes.text.registry)

graph = Graph.compile(make_graph(conductor_nodes.text.split, conductor_nodes.text.uppercase), registry)
print(graph.run(text="a\nb").outputs)  # {'result': ['A', 'B']}
```

| Module | Category | Nodes |
|---|---|---|
| `text` | `text` | `text-uppercase`, `text-lowercase`, `text-trim`, `text-length`, `text-concat`, `text-replace`, `text-contains`, `text-split`, `text-list`, `text-join`, `text-reverse` |
| `math` | `math` | `math-add`, `math-subtract`, `math-multiply`, `math-divide`, `math-modulo`, `math-round`, `math-min`, `math-max`, `math-abs` |
| `logic` | `logic` | `logic-not` |
| `json_ops` | `json` | `json-parse`, `json-stringify`, `json-get` |
| `regex_ops` | `regex` | `regex-match`, `regex-replace`, `regex-extract` |
| `series` | `series` | `series-split`, `series-collect`, `series-stack` |
| `tables` | `table` | `table-collect` |
| `control` | `control` | `control-decide`, `control-if-empty`, `control-if-equals` |

Each module is one category, so a module's `registry` and a filter by category pick the same nodes: `conductor_nodes.registry.filter(lambda d: d.category == "math")`.

`series-split` takes a table or a list apart, so you work through it one record or item at a time. Everything after it runs once for each, and `table-collect` puts the records back into one table, with a new column beside them for what was returned for each:

```python
import io

import polars as pl
from conductor import Table
from conductor_nodes.series import split
from conductor_nodes.tables import collect

registry.add_subregistry(conductor_nodes.series.registry)
registry.add_subregistry(conductor_nodes.tables.registry)


@registry.node("read-contacts", title="Read contacts", description="Reads contacts from a CSV text.")
def read_contacts(csv: str) -> Table:
    return pl.read_csv(io.StringIO(csv))


@registry.node("greet", title="Greet", description="Greets a person by name.")
def greet(name: str) -> str:
    return f"Hello {name}"


contacts = Graph.compile(make_graph(read_contacts, split, greet, collect, edges=[
    ("read-contacts.result", "series-split.value"),
    ("series-split.result.name", "greet.name"),
    ("series-split.result", "table-collect.records"),
    ("greet.result", "table-collect.greeting"),
], constants={"table-collect.columns": ["greeting"]}, outputs={"contacts": "table-collect.result"}), registry)
print(contacts.run(csv="name\nIda\nBo\n").outputs["contacts"].to_dicts())  # [{'name': 'Ida', 'greeting': 'Hello Ida'}, {'name': 'Bo', 'greeting': 'Hello Bo'}]
```

- `series-split.result.name` is one field of each record. `table-collect` works out its output when the graph is compiled, from `columns`, so `make_graph` is told the graph returns it: `outputs={"contacts": "table-collect.result"}`. `greet` takes one name, so it runs once per contact.
- `table-collect` collects the loop: it is called once, with every record and every greeting. Each name in `columns` is an input taking one value per record, `None` where a record got none.
- Each name in `constant_columns` is an input taking one value from outside the loop, an outer value, written on every record: the file's name on each line.
- The two are a pair. A value returned per record goes in `columns`; a value returned once goes in `constant_columns`. A node that runs once wired into a name in `columns` has one value for all the records, and the run fails with `columns_not_aligned`, which names `constant_columns` for a column that has one value.

When a whole table or list is wired to an input that takes one value, compiling reports `type_mismatch` with `ctx.variant` `make_a_series`, and an edge that names a field of one reports `unknown_output` with the same variant: put `series-split` between them. `series-split` takes the type that arrives: a table of contacts returns contacts, a list of numbers returns numbers, and anything else is a `type_mismatch` on the edge. `series-stack` goes the other way for tables: where a node returns a table per item, say the lines of each invoice, it puts them into one table, one under the other.

`series-collect` is the way back to one list: it is called once and returns the value of every item, in order.

`text-split` splits a text into its lines, or at another separator. `text-list` returns the texts set in its constant; wire a series to `for_each` and it returns them once for each item of it, so each document gets every question.

`control-decide` gives its `value` on `if_true` or `if_false` by `when`, and the nodes wired to the other output are skipped for that call. Its outputs take the type that arrives on `value`:

```python
from conductor_nodes.control import decision

graph = Graph.compile(make_graph(conductor_nodes.text.length, decision, edges=[("text-length.result", "control-decide.value")]), conductor_nodes.registry)
print([field.type for field in graph.describe().nodes["control-decide"].outputs])  # ['int', 'int']
```

A JSON value is pydantic's `JsonValue`. The regular-expression nodes run on the `regex` package, with a timeout on every call, so a pattern that would freeze `re` fails its node as `pattern_timeout`; one that doesn't compile fails as `invalid_pattern`, and a replacement that refers to a group the pattern lacks as `invalid_replacement`. `math-round` rounds a half away from zero, 2.5 to 3 and 0.125 to 0.13 at two decimals.

The documentation, the examples and the changelog live in the repository: https://github.com/syv-ai/conductor
