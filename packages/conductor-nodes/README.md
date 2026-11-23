# syv-conductor-nodes

Ready-made nodes for [Conductor](https://github.com/syv-ai/conductor). They work on text, numbers, yes or no, JSON, regular expressions, lists and tables, and they send a value one way or another. None of them call out to anything: they only work on what they're given.

```bash
uv add syv-conductor-nodes
```

## Add them to your registry

`conductor_nodes.registry` has every node. Each module has a `registry` of its own too, so you can add only the ones you want:

```python
import conductor_nodes
from conductor import Graph, NodeRegistry, make_graph

registry = NodeRegistry()
registry.add_subregistry(conductor_nodes.text.registry)

graph = Graph.compile(make_graph(conductor_nodes.text.split, conductor_nodes.text.uppercase), registry)
print(graph.run({"text": "a\nb"}).outputs)  # {'result': ['A', 'B']}
```

Each module is one category, so filtering by category gives the same nodes: `conductor_nodes.registry.filter(lambda d: d.category == "math")`.

## The nodes

Each module holds one kind of node. The category is what `describe()` shows for its nodes:

- `text` (category `text`): change, split, join and search text.
- `math` (category `math`): add, divide, round and other arithmetic.
- `logic` (category `logic`): answer yes or no about a value.
- `control` (category `control`): send a value one way or another.
- `json_ops` (category `json`): read and write JSON.
- `regex_ops` (category `regex`): match, replace and find with regular expressions.
- `series` (category `series`): start and end a loop.
- `tables` (category `table`): put the rows of a loop back into a table.

To see every node with its title, loop over the registry:

```python
for definition in conductor_nodes.registry:
    print(definition.id, definition.title, sep=": ")
```

## Loops: one row at a time

`series-split` takes a table or a list apart. Every node after it runs once for each row or item. `table-collect` then puts the rows back into one table, with a new column for what was worked out for each.

Here we read some contacts, greet each one, and put the greetings next to the names:

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


contacts = Graph.compile(
    make_graph(
        read_contacts,
        split,
        greet,
        collect,
        edges=[
            ("read-contacts.result", "series-split.value"),
            ("series-split.result.name", "greet.name"),
            ("series-split.result", "table-collect.rows"),
            ("greet.result", "table-collect.greeting"),
        ],
        constants={"table-collect.columns": ["greeting"]},
        outputs={"contacts": "table-collect.result"},
    ),
    registry,
)
greeted = contacts.run({"csv": "name\nIda\nBo\n"}).outputs["contacts"]
assert isinstance(greeted, pl.DataFrame)
print(greeted.to_dicts())  # [{'name': 'Ida', 'greeting': 'Hello Ida'}, {'name': 'Bo', 'greeting': 'Hello Bo'}]
```

A few things to notice:

- `series-split.result.name` is the `name` field of each row. `greet` takes one name, so it runs once per contact.
- `columns` names the new columns. Each name becomes an input of `table-collect`, here `greeting`, with one value per row.
- `table-collect` works out its output when the graph is compiled, so `make_graph` needs to be told the graph returns it, with `outputs=`.
- For a value that's the same on every row, like the name of the file the contacts came from, use `constant_columns` instead.

If you wire a whole table or list into an input that takes one value, compiling reports a `type_mismatch`. It asks for a node in between that splits the table or list up, and `series-split` is that node.

`series-collect` is the simple way back: it returns the value of every item as one list. `series-stack` is for a loop that gives a table per item, say the lines of each invoice. It stacks them into one table.

How loops work in full: `python -m conductor.about one-or-many`.

## Branches

`control-decide` sends its `value` to `if_true` or `if_false`, by `when`. Each is a branch: an output whose default is `SKIPPED`. The nodes after the other output are skipped. `when` comes from a node that answers yes or no, such as `logic-equals`, `logic-is-empty` or `text-contains`. `when` has no default, so a graph that wires nothing to it and gives it no value is `input_has_no_value` when it is compiled.

`control-switch` has one output per name in its `cases`, and one more called `other`. You can wire several branches into one input, and that node takes whichever one has the value.

`control-default` gives its `value`, or its `fallback` when the value is `None`.

The outputs take the type of whatever is wired to `value`:

```python
from conductor_nodes.control import decision

graph = Graph.compile(make_graph(conductor_nodes.text.length, decision, edges=[("text-length.result", "control-decide.value")]), conductor_nodes.registry)
print([field.type for field in graph.describe().nodes["control-decide"].outputs])  # ['int', 'int']
```

## Good to know

- The regular-expression nodes use the `regex` package, and every search has a timeout. The `regex` package finishes many patterns that would make Python's `re` hang, and one that still runs too long fails the node with `pattern_timeout`.
- `math-round` rounds a half away from zero, the way people do: 2.5 becomes 3, and 0.125 becomes 0.13 at two decimals.
- Every node is marked `Policy(costs_nothing=True)`. So `run.resume(only_costs_nothing=True)` calls them, and a host can count the calls after a `series-split` before it calls anything that costs.
- `json-parse` and `json-stringify` refuse a number that isn't finite, with `not_json`. `json-parse` refuses `NaN`, `Infinity` and `-Infinity`, which aren't JSON, and a number too large for a float, such as `1e999`. `json-stringify` refuses a value holding `inf` or `nan`, such as a `math-multiply` result too large for a float.
- When a node can't do its job, it fails with a code you can act on, such as `division_by_zero`, `not_json` or `invalid_pattern`.

The documentation, the examples and the changelog are in the repository: https://github.com/syv-ai/conductor
