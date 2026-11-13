# Design

## One rule, one place

Write each rule, grammar, format or fact once and call it from every other site; this is about knowledge, not about every repeated line. A text's reader takes only what its writer writes, and a rule that holds for one kind of node is one function for every kind, not a second copy with an exemption.

Why: DRY means each piece of knowledge has one authoritative place, not that every repeated line is abstracted.

```python
gone = set(older.input_names) - set(newer.input_names)   # bad: in upgrade.py, while declare.py skips a subgraph
gone = gone_inputs(older, newer)                          # good: the declaration and the upgrade call one rule
```

## No mirrors

Never keep two structures in step with a test, a comment or a convention; derive one from the other so there is one source.

Why: two copies kept equal by a test drift sooner or later; deriving one from the other removes the drift.

```python
assert listed_names("llms.txt") == set(conductor.__all__)   # bad: a test holds two copies equal
# good: the reference writes its list of names from __all__ when it is read
```

## Look up, don't scan

Build a keyed form once and look values up by key; never loop over the same collection several times or count and search a list once per item.

Why: a lookup by key says what is wanted and costs the same at any size; a scan says neither.

```python
[i for i in graph_inputs if names.count(i.name) > 1]                 # bad
counts = Counter(names); [i for i in graph_inputs if counts[i.name] > 1]   # good
```

## Readability over speed

Never add a shortcut, second structure or repeated line for speed unless a run would otherwise be slower than conductor's last release, and never for work that release does not do.

Why: conductor is already faster than its 1.x line; between readable and faster, readable wins.

```python
if shareable:  # speed only: the extra pass is skipped    # bad
schema = _with_shared_tables(schema)                       # good
```

## The wire is never an argument

Judge a design only by what it simplifies in conductor, never by the JSON or names a host sees, and turn a shorter typed form into the one model at its door.

Why: conductor owns its JSON and its one host can change with it, so a shape is kept only if it makes conductor simpler.

```text
Bad:  Keep a source as "node.output" text, because hosts write it.
Good: make_graph reads the text once; the record holds a Source.
```

## Tests read public names

Pin behaviour in a test, its ids and its helpers through names that are not private, and never read or patch private state such as `run._outcomes`.

Why: a test that reads private state pins today's design and breaks on every refactor that keeps the behaviour.

```python
assert run._outcomes[("shout",)][0].outputs["result"] == "HI"   # bad
assert run.results["shout"]["result"] == "HI"                    # good
```
