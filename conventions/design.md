# Design

## One rule, one place

Write each rule, grammar, format or fact once and call it from every other site; this is about knowledge, not about every repeated line.

Why: "DRY doesn't mean abstract away any repeat code, it is literally 'Every piece of knowledge must have a single, unambiguous, authoritative representation within a system.'"

```python
digest = blake2b(dumps(value, sort_keys=True).encode()).hexdigest()   # bad: a fourth recipe
digest = json_hash(value, 8, keys_sorted=True)                        # good: the one way
```

## No mirrors

Never keep two structures in step with a test, a comment or a convention; derive one from the other so there is one source.

Why: "As a general rule: we don't do mirrors and twins pinned by tests. It is bound to drift somehow. Fix the cause upstream."

```python
assert listed_names("llms.txt") == set(conductor.__all__)   # bad: a test holds two copies equal
# good: the reference writes its list of names from __all__ when it is read
```

## Look up, don't scan

Build a keyed form once and look values up by key; never loop over the same collection several times or count and search a list once per item.

Why: "Any place we can make direct lookups, dict-style, is much preferred."

```python
[i for i in graph_inputs if names.count(i.name) > 1]                 # bad
counts = Counter(names); [i for i in graph_inputs if counts[i.name] > 1]   # good
```

## Readability over speed

Never add a shortcut, second structure or repeated line for speed unless a run would otherwise be slower than conductor's last release, and never for work that release does not do.

Why: "If we have to pick between readability/maintainability and speed improvements, we are picking the former."

```python
if shareable:  # speed only: the extra pass is skipped    # bad
schema = _with_shared_tables(schema)                       # good
```

## The wire is never an argument

Judge a design only by what it simplifies in conductor, never by the JSON or names a host sees, and turn a shorter typed form into the one model at its door.

Why: "We don't care about the wire! … This should never be an argument."

```text
Bad:  Keep a source as "node.output" text, because hosts write it.
Good: make_graph reads the text once; the record holds an Address.
```

## Tests read public names

Pin behaviour in a test, its ids and its helpers through names that are not private, and never read or patch private state such as `run._outcomes`.

Why: "We are not in the business of fitting code to tests."

```python
assert run._outcomes[("shout",)][0].outputs["result"] == "HI"   # bad
assert run.results["shout"]["result"] == "HI"                    # good
```
