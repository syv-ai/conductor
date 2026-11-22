# Types and models

## Fail loud

Let an unexpected state raise: no `except Exception: return None`, no silent default and no fallback that turns a fault into a skip.

Why: a silent default turns a bug into a wrong result found much later; a loud failure points at the cause.

```python
try: return to_json(value)
except Exception: return None    # bad
return to_json(value)            # good
```

## Invalid states cannot be written

Split a record whose fields hide an either/or into two types, give no field a `None` or empty value in one of its uses or a union of several types for one thing, and no type two meanings (a path to a node or to an output).

Why: a union of several shapes for one field means the shape was not decided where the value is made.

```python
class Version: function: Callable | None; graph: GraphRecord | None   # bad
class FunctionVersion: function: Callable                            # good, with
class SubgraphVersion: graph: GraphRecord
```

## What a person sees is declared

Read text a person sees from what `@node`, `@host_type` or `Param(title=...)` declares, never guess it from a class name, a docstring or `getattr` with a default, and advise in a problem only what is right in every case.

Why: what a person sees must never be guessed from a code name; it is declared where the node or type is.

```python
title = getattr(cls, "title", None) or cls.__name__   # bad
@host_type("money", title="Amount of money")          # good
```
