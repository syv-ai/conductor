# Validation

## Validation is the wall

Validate untyped outside data once, with pydantic, at its door in `_internal/gate/` (JSON loaded, values a caller gives a run, what a node returns, a person's answer), and never build a validator or check a value again anywhere else.

Why: "Think of conductor as a fortress and the validation is our wall … Everyone that passes through that gate is frisked thoroughly … So there's no reason to stop them and ask everytime we see them."

```python
if not is_name(record.id): problems.append(...)   # bad: in compiled/compile.py
# good: the record door in gate/record.py finds it; compile trusts the record
```

## Text is parsed once, at its door

Parse an address with `Address.parse` where the text enters and keep the `Address`; never split, join or re-parse address text anywhere else.

Why: "I also see a lot of address wrangling and checks outside the address.py module."

```python
node, name = target.split(".", 1)    # bad
address = Address.parse(target)      # good
```
