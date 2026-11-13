# Principles

They come before every other file here; where a convention and a principle pull apart, the principle wins.

## Least astonishment, KISS, single responsibility

Give each module, class and function one job, put it where a reader would look for it, and choose the simplest code that does the job.

Why: code a reader can guess, built from the simplest shape that works, with one job per piece, is code that stays easy to change.

```text
Bad:  make_graph, a helper on top of everything, sits beside ConductorModel, which everything stands on.
Good: base/model.py holds ConductorModel; graph/make.py holds make_graph.
```

## Deep modules, uniform surfaces

Give every concept package the same layout and one small surface the engine calls the same way, and never make the engine branch on which kind of thing it holds.

Why: a narrow interface over a deep module hides its complexity, and one shape across packages means a reader learns it once.

```python
if type(outcome) is Skipped or type(outcome) is NoneInput: ...   # bad: the caller asks which kind
match outcome:                                                   # good: an exhaustive match where kinds differ
    case Done() | Answered() | Unanswered(): ...
    case Skipped() | NoneInput(): ...
    case _: assert_never(outcome)
```

## Fewer concepts and models

Do not add a class, model, protocol, alias or sentinel when an existing one holds the concept, and never a second shape of one (keyed against list, checked against raw); if a new one is a step towards removing more, say which in the commit message.

Why: every concept is something each reader must learn; a second shape of one concept (keyed beside list, checked beside raw) adds one without removing any.

```python
class KeyedGraphRecord(BaseModel): nodes: dict[str, NodeRecord]   # bad: a keyed twin of GraphRecord
class GraphRecord(BaseModel): nodes: dict[str, NodeRecord]        # good: the one record, keyed by id
```

## Change radically when it simplifies

Rewrite, move or delete existing code and its tests whenever the result is simpler; existing code and tests are never a reason to keep a shape.

Why: conductor's own code and tests decide nothing; a change that leaves fewer concepts is worth its size.

```text
Bad:  Keep Graph.load because forty tests call it.
Good: Rename it to Graph.compile and change the forty tests.
```
