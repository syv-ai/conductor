# Types and models

## Trust the declared type

Never re-check at run time what a parameter's annotation already declares (no `isinstance` guard, `_raise_unless_...` or `checked_...`); when a value's type is uncertain, trace it back to where it enters and make it certain there.

Why: "We HATE defensive programming. We want to fail loudly if something that we didn't expect to happen happens. We also despise redundant None/type/instance checks." "When we construct a NodeDefinition we should 100% know that it's a NodeDefinition."

```python
def add(self, definition: NodeDefinition) -> None:
    if not isinstance(definition, NodeDefinition): raise NodeDeclarationError(...)   # bad: delete the check
```

## Fail loud

Let an unexpected state raise: no `except Exception: return None`, no silent default and no fallback that turns a fault into a skip.

Why: "We want to fail loudly if something that we didn't expect to happen happens."

```python
try: return to_json(value)
except Exception: return None    # bad
return to_json(value)            # good
```

## Invalid states cannot be written

Split a record whose fields hide an either/or into two types, and give no field a `None` or empty value in one of its uses or a union of several types for one thing.

Why: "I see a union of 4 types allowed for field `widget`. Why? That to me indicates a fragmentation upstream."

```python
class Version: function: Callable | None; graph: GraphRecord | None   # bad
class FunctionVersion: function: Callable                            # good, with
class SubgraphVersion: graph: GraphRecord
```

## What a person sees is declared

Read text a person sees from what `@node`, `@host_type` or `Param(title=...)` declares, and never guess it from a class name, a docstring or `getattr` with a default.

Why: "There should be absolutely *NO* question what the user sees."

```python
title = getattr(cls, "title", None) or cls.__name__   # bad
@host_type("money", title="Amount of money")          # good
```
