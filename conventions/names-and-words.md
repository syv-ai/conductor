# Names and words

Names, docstrings and comments.

## Plain words a stranger reads right

Write every name, docstring and comment so a Python developer who has never seen conductor or its plans reads it right: plain Python words, graph words where one fits, no coined metaphors and no word taken from a plan or word list.

Why: "The language and variable naming is incredibly esoteric." "If we cannot explain it simply, we don't understand it well enough." "There are no settled vocabulary. If there is, do not trust it."

```python
"""Going on is one leg of the run, as it stands."""                                   # bad
"""Whether the run can go on: it has calls left to make and nobody has cancelled it."""  # good
```

## Names the owner ruled out

Never use these names, and rename one on any line you edit: `WrongAsk`, `Referred`, `Placed`/`placed`, `_Said`/`_say`, `_places`, `_inlined`, `listed()`, `comes_to()`, `wanted`, `leg`/`_leg`, `EarlierWork`, `_as_earlier_work`, `earlier_columns`, `FunctionAtPath`, `EmbeddedGraphAtPath`, `at_path`, `node_at`, a parameter `at`, `AsItStandsReader`, "as it stands", `SavedForm`, `has_every_unit()`, a package `saved_run`, "birthed", "barren"; `_hint` only for a type hint, "sentinel" never "mark", and inputs "are" a value, never "hold" it.

Why: "WrongAsk is a pretty bad name." "If we cannot explain it simply, we don't understand it well enough."

```python
wanted_args, held_args = get_args(target), get_args(source)      # bad
expected_args, given_args = get_args(target), get_args(source)   # good
```

## Conductor stands alone

Write no host's words (flow, app), imports, access model or concrete types into conductor, and change it for a host's finding only when it holds as a general library need.

Why: "conductor shouldn't be aware of host types at all. The idea is that nodes a built on the host's side." "We would generally prefer to stay in graph terminology whenever possible."

```python
def run_flow(flow: FlowRecord) -> None: ...      # bad
def run_graph(record: GraphRecord) -> None: ...  # good
```

## Errors end in Error

Name every exception class for what went wrong and end it in `Error`; `Ask` is the one exception.

Why: "it should have an Error suffix so we know this is an actual error (not a hack of errors like Ask is)."

```python
class WrongAsk(Exception): ...         # bad
class InvalidAskError(Exception): ...  # good
```

## Google-style docstrings

Give every public class and function a Google-style docstring that says what it does, with `Args`, `Returns`, `Raises` and `Attributes` saying what each argument and field holds.

Why: "In general, the docstrings are a mess. Use Google's format consistently in Conductor." "The docstring needs to explain what it does and what the fields do."

```python
def read_answer(waiting, answer):
    """Reads it."""                                       # bad

def read_answer(waiting: Waiting, answer: Any) -> Answer:
    """Reads a person's answer into the fields it fills.

    Args:
        waiting: The call waiting for the answer.
        answer: The answer as the host received it.
    """                                                   # good
```

## No history in code

Write code, comments, docstrings and docs in the present tense of what the code does, with no "renamed from", "formerly", "no longer", dates, plan files, decision ids or PR numbers; history goes only in `CHANGELOG.md`.

Why: "We do not want ANY temporal comments or reference to any iterations or past states outside CHANGELOG — and even there, we use it sparingly."

```toml
# The order of the module map (plans: D-L143), with its third row split.        # bad
# Packages on one line with " | " between them import none of each other.       # good
```
