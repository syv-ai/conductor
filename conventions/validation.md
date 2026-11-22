# Validation

## Validation is the wall

Validate untyped outside data once, with pydantic, where it enters, in a validator in the `validator.py` of the concept it is a value of (JSON loaded, values a caller gives a run, what a node returns, a person's answer), checking with the checks the value's `Type` builds (`argument_validator`, `output_validator`); build a type's checks in its type's package (`types/`, `table/`, `series/`), put a model's own invariant on the model as a pydantic validator, check a value again inside only where an edge's input demands more than its source declared, and wrap only the validator's call in a caller's `try`, catching its `InvalidCallError`.

Why: a value checked once where it enters can be trusted everywhere inside, so checks inside are noise that hides the real ones.

```python
if not is_name(node_id): problems.append(...)   # bad: in graph/compile.py
# good: the record validator in record/validator.py finds it; compile trusts the record
```

## A developer's mistake is caught before a run

Check on every call only what code can't rule out: a caller's or a person's value at its door, and what a node returns, even a value its code only hands on from elsewhere. Catch a mistake in code a developer wrote (a node's body or declaration, a host type, a hook, conductor's own code) before a run: by its types, at `@node` or `@host_type`, when compiling, or in `check_registry`. Keep a per-call check only where data decides when the mistake happens, or where it would otherwise go silently wrong. Never guard a declared parameter with `isinstance`: when a value's type is uncertain, trace it back to where it enters and make it certain there.

Why: a mistake in code is fixed once, by changing the code, so a check paid on every call costs every run in production and catches nothing after the fix, and a guard for a state the types rule out hides the bug that made it.

```python
if arguments != before: raise ...   # bad: in run/call.py, after every call, for a body that changes its inputs
# good: check_registry calls each node with made-up values and reports changed_its_inputs
```

## A validator is a class in its concept's validator.py

Read each kind of outside value with a subclass of `Validator` in the `validator.py` of the concept it is a value of, built with what it reads the value by and called with the value alone.

Why: one shape and one place for every validator, so a reader finds each where its concept is and calls it the same way.

```python
read_constant(node_id, port, value)        # bad: a function with a signature of its own
ConstantValidator(node_id, port)(value)    # good: in graph/validator.py
```

## Text is read once, where it enters

Read text once where it enters and keep what it gives (JSON through `read_json`, a dotted place only in `make_graph`), never split, join or read it again, and write a place as text only with `place_text`.

Why: text split or parsed in several places has several spellings and several sets of faults.

```python
node, output = target.split(".", 1)          # bad: in graph/compile.py
node, output = source.node, source.output    # good: the record holds a Source
```
