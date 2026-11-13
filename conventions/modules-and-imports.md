# Modules, packages and imports

## A concept is a package

Put each concept of `_internal/` in its own package, its surface in `__init__.py` with `__all__`, and name its files by job (`type.py` the class, `read.py` its schema for untyped data, `write.py` its JSON form, `stored.py` how a run stores it, `describe.py` what a host is shown) or by the noun they are about.

Why: a few large files mix many jobs; a package per concept lets each module have one reason to change.

```text
Bad:  _internal/node_definition.py holds the declaration, the definition, its policy and its palette form.
Good: _internal/node/declare.py, type.py, policy.py, describe.py.
```

## Imports go through a package's surface

Import another package only through its `__init__.py` and only from a lower layer, your own package's modules by a relative name, never `*`, a name with a leading underscore from another module or an attribute with one from another package, and register nothing at import.

Why: a module that reaches into another package's files ties the two together and mixes layers.

```python
from conductor._internal.run.outcome import Done   # bad: a file inside another package
from conductor._internal.run import Done           # good; inside run/: from .outcome import Done
```

## JSON in and out is one concept

Keep reading and writing JSON apart from the class it is about (a run's JSON form in `run/json/`), and write no second JSON reader or writer beside it.

Why: JSON read or written in several places is several rules that can disagree.

```python
def from_json(text: str) -> dict[str, Any]: ...         # bad: in gate/answer.py
def read_as_answer(value: Any) -> dict[str, Any]: ...   # good: the door reads a value
```

## Classes are imported by name

Import a class or exception by name and write it bare, and use `module.` only for functions and constants (exceptions in `tests/test_core/test_imports_by_name.py`).

Why: a class written through its module reads longer and hides which names a file depends on.

```python
except json.JSONDecodeError: ...                                # bad
from json import JSONDecodeError; except JSONDecodeError: ...   # good
```

## No helper without a job

Fold a helper with one caller and no work of its own into its caller, and use the standard library or a stock tool before writing your own.

Why: a helper that only renames a call, or wraps one line, is one more name to learn and nothing to gain.

```python
def _name_of_route(route): return route.name   # bad
generate_unique_id_function=attrgetter("name")  # good
```
