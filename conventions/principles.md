# Principles

They come before every other file here; where a convention and a principle pull apart, the principle wins.

## Least astonishment, KISS, single responsibility

Give each module, class and function one job, put it where a reader would look for it, and choose the simplest code that does the job.

Why: "Your primary guiding principles are Least Astonishment, KISS and Single Responsibility."

```text
Bad:  make_graph, a helper on top of everything, sits beside ConductorModel, which everything stands on.
Good: base/model.py holds ConductorModel; graph/make.py holds make_graph.
```

## Deep modules, uniform surfaces

Give every concept package the same layout and one small surface the engine calls the same way, and never make the engine branch on which kind of thing it holds.

Why: "Your prime directive is deep modules, uniform surfaces, even at the cost of some ceremony." "We *also* want one interface that the module offers the engine."

```python
if type(outcome) is Skipped or type(outcome) is NoneInput: ...   # bad: the caller asks which kind
match outcome:                                                   # good: an exhaustive match where kinds differ
    case Done() | Answered() | Unanswered(): ...
    case Skipped() | NoneInput(): ...
    case _: assert_never(outcome)
```

## Fewer concepts and models

Do not add a class, model, protocol, alias or sentinel when an existing one holds the concept, and never a second shape of one (keyed against list, checked against raw); if a new one is a step towards removing more, say which in the commit message.

Why: "The worst thing that can happen is that the number of concepts and models grow (unless it's transitional to remove a higher number of them later)." "KeyedGraphRecord vs GraphRecord is the prime examples. It's piling on, not redesigning in a better way."

```python
class KeyedGraphRecord(BaseModel): nodes: dict[str, NodeRecord]   # bad: a keyed twin of GraphRecord
class GraphRecord(BaseModel): nodes: dict[str, NodeRecord]        # good: the one record, keyed by id
```

## Change radically when it simplifies

Rewrite, move or delete existing code and its tests whenever the result is simpler; existing code and tests are never a reason to keep a shape.

Why: "Do NOT be afraid to make radical changes to existing code as long as it simplifies." "We are not in the business of fitting code to tests."

```text
Bad:  Keep Graph.load because forty tests call it.
Good: Rename it to Graph.compile and change the forty tests.
```
