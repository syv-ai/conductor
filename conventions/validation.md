# Validation

## Validation is the wall

Validate untyped outside data once, with pydantic, where it enters, in a validator in the `validator.py` of the concept it is a value of (JSON loaded, values a caller gives a run, what a node returns, a person's answer), never build a validator or check a value again anywhere else, and wrap only the validator's call in a caller's `try`, catching its `InvalidCallError`.

Why: a value checked once where it enters can be trusted everywhere inside, so checks inside are noise that hides the real ones.

```python
if not is_name(node_id): problems.append(...)   # bad: in graph/compile.py
# good: the record validator in record/validator.py finds it; compile trusts the record
```

## Text is read once, where it enters

Read text once where it enters and keep what it gives (JSON through `read_json`, a dotted place only in `make_graph`), never split, join or read it again, and write a place as text only with `place_text`.

Why: text split or parsed in several places has several spellings and several sets of faults.

```python
node, output = target.split(".", 1)          # bad: in graph/compile.py
node, output = source.node, source.output    # good: the record holds a Source
```
