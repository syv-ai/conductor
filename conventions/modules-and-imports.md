# Modules, packages and imports

## A concept is a package

Put each concept of `_internal/` in its own package, its surface in `__init__.py` with `__all__`, and name its files by job (`type.py` the class, `read.py` its schema for untyped data, `write.py` its JSON form, `stored.py` how a run stores it, `describe.py` what a host is shown) or by the noun they are about.

Why: "We currently stay within a few files like our lives depend on it. There is no reason to. Spread out into reasonable modules; observe single responsibility principle."

```text
Bad:  _internal/node_definition.py holds the declaration, the definition, its policy and its palette form.
Good: _internal/node/declare.py, type.py, policy.py, describe.py.
```

## Imports go through a package's surface

Import another package only through its `__init__.py` and only from a lower layer, your own package's modules by a relative name, never `*`, a name with a leading underscore from another module or an attribute with one from another package, and register nothing at import.

Why: "We also mix layers a fair bit inside _internal."

```python
from conductor._internal.run.outcome import Done   # bad: a file inside another package
from conductor._internal.run import Done           # good; inside run/: from .outcome import Done
```

## JSON in and out is one concept

Keep reading and writing JSON apart from the class it is about (a run's JSON form in `run/json/`), and write no second JSON reader or writer beside it.

Why: "Why is there a `from_json` in _internal/gate/answer.py? We said that IO should be consolidated."

```python
def from_json(text: str) -> dict[str, Any]: ...         # bad: in gate/answer.py
def read_as_answer(value: Any) -> dict[str, Any]: ...   # good: the door reads a value
```

## Classes are imported by name

Import a class or exception by name and write it bare, and use `module.` only for functions and constants (exceptions in `tests/test_core/test_imports_by_name.py`).

Why: the owner named `except pydantic_core.PydanticSerializationError:` as a habit to drop.

```python
except json.JSONDecodeError: ...                                # bad
from json import JSONDecodeError; except JSONDecodeError: ...   # good
```

## No helper without a job

Fold a helper with one caller and no work of its own into its caller, and use the standard library or a stock tool before writing your own.

Why: "a lot of places where helper functions are completely redundant." "we can wrap itemgetter for this."

```python
def _name_of_route(route): return route.name   # bad
generate_unique_id_function=attrgetter("name")  # good
```
