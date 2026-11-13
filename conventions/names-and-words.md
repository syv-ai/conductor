# Names and words

## Plain words a stranger reads right

Write every name, docstring and comment so a Python developer who has never seen conductor or its plans reads it right: plain Python words, graph words where one fits, no coined metaphors and no word taken from a plan or word list.

Why: a name or a sentence that needs the plans to be understood is not understood; if it cannot be said simply, the idea is not clear yet.

```python
"""Going on is one leg of the run, as it stands."""                                   # bad
"""Whether the run can go on: it has calls left to make and nobody has cancelled it."""  # good
```

## Ruled-out names

Never use these names, and rename one on any line you edit: `WrongAsk`, `Referred`, `Placed`/`placed`, `_Said`/`_say`, `_places`, `_inlined`, `listed()`, `comes_to()`, `wanted`, `leg`/`_leg`, `EarlierWork`, `_as_earlier_work`, `earlier_columns`, `FunctionAtPath`, `EmbeddedGraphAtPath`, `at_path`, `node_at`, a parameter `at`, `AsItStandsReader`, "as it stands", `SavedForm`, `has_every_unit()`, a package `saved_run`, "birthed", "barren"; `_hint` only for a type hint, "sentinel" never "mark", and inputs "are" a value, never "hold" it.

Why: these names were read wrong or said nothing; a plain name says the job.

```python
wanted_args, held_args = get_args(target), get_args(source)      # bad
expected_args, given_args = get_args(target), get_args(source)   # good
```

## Conductor stands alone

Write no host's words (flow, app), imports, access model or concrete types into conductor, and change it for a host's finding only when it holds as a general library need.

Why: conductor is a library for any host; a host's words or types in it fit it to one application.

```python
def run_flow(flow: FlowRecord) -> None: ...      # bad
def run_graph(record: GraphRecord) -> None: ...  # good
```

## Errors end in Error

Name every exception class for what went wrong and end it in `Error`; `Ask` is the one exception.

Why: the suffix tells a reader the class is a real error, unlike `Ask`, which is raised to ask a person.

```python
class WrongAsk(Exception): ...         # bad
class InvalidAskError(Exception): ...  # good
```

## Google-style docstrings

Give every public class and function a Google-style docstring that says what it does, with `Args`, `Returns`, `Raises` and `Attributes` saying what each argument and field is for.

Why: one docstring format, saying what a thing does and what each field means, is what a reader and the reference read.

```python
def read_answer(waiting, answer):
    """Reads it."""                                       # bad

def read_answer(waiting: Waiting, answer: Any) -> dict[str, Any]:
    """Reads a person's answer into the outputs of the node that asked.

    Args:
        waiting: The call waiting for the answer.
        answer: The answer as the host received it.
    """                                                   # good
```

## No history in code

Write code, comments, docstrings and docs in the present tense of what the code does, with no "renamed from", "formerly", "no longer", dates, plan files, decision ids or PR numbers; history goes only in `CHANGELOG.md`.

Why: code says what it does now; its history lives in git and, sparingly, in CHANGELOG.

```toml
# The order of the module map (plans: D-L143), with its third row split.        # bad
# Packages on one line with " | " between them import none of each other.       # good
```
