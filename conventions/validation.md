# Validation

## Validation is the wall

Validate untyped outside data once, with pydantic, at its door in `_internal/gate/` (JSON loaded, values a caller gives a run, what a node returns, a person's answer), never build a validator or check a value again anywhere else, and wrap only the door call in a caller's `try`, catching its `InvalidCallError`.

Why: "Think of conductor as a fortress and the validation is our wall … Everyone that passes through that gate is frisked thoroughly … So there's no reason to stop them and ask everytime we see them."

```python
if not is_name(node_id): problems.append(...)   # bad: in compiled/compile.py
# good: the record door in gate/record.py finds it; compile trusts the record
```

## Text is read once, at its door

Read text once where it enters and keep what it gives (JSON through `read_json`, a dotted place only in `make_graph`), never split, join or read it again, and write a place as text only with `place_text`.

Why: "I also see a lot of address wrangling and checks outside the address.py module."

```python
node, output = target.split(".", 1)          # bad: in compiled/compile.py
node, output = source.node, source.output    # good: the record holds an Address
```
