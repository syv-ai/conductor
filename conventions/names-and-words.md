# Names and words

## Plain words a stranger reads right

Write every name, docstring and comment for a Python developer who has never seen conductor. Say what the thing is first, in words they already know. Then say where it fits and why it exists. Never assume they know the rest of the codebase: someone who opens one file should understand it from that file. Use plain Python and graph words, with no coined metaphors and no word taken from a plan. Use the plain verb for what the code does: check, convert or store, not "read" for all three, "the form a run keeps" or "holds to". A problem's message is written in the same voice.

Why: a sentence that only makes sense once you know the rest of the code isn't understood. If it can't be said simply, the idea isn't clear yet.

```python
"""The question about each output as a call asks it, and as a palette shows the output."""   # bad

"""The questions a person is shown when a node asks.

A node that raises ``Ask`` is answered by filling in its outputs, and each output is one question.
"""                                                                                            # good
```

## One idea per sentence

Write short sentences with one idea each, in the tone you would use to explain the code to a colleague. Don't fit every fact into one sentence: split it. Say each thing once, where a reader looks for it: the package docstring says what the concept is and where it fits, the class teaches how to use it, and a helper says what it checks. Leave out what the code says, and point to the reference (`python -m conductor.about <section>`) instead of repeating its rules. Where it helps, add an example with real values.

Why: a sentence that carries five facts has to be read three times. Short, relaxed sentences are read once.

```python
"""Reads ``value`` as a person's answer of ``value_type`` is read, into the form a run keeps."""   # bad

"""Reads one value as a person's answer of type ``value_type``.

An answer is read strictly, the way its JSON text would be read. So ``True`` is a ``bool``, but
``"yes"`` is refused.
"""                                                                                                 # good
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

Give every public class and function a Google-style docstring, and hold a private helper's docstring to the same rule. Its first line says what the thing is or does, in a short sentence a reader gets in one go. A few plain sentences follow where they help, then `Args`, `Returns`, `Raises` and `Attributes`, each saying what an argument or field is for.

Why: one format is easy to read and to find things in. The reference shows a public name's first paragraph alone, so that paragraph has to make sense on its own.

```python
class AnswerValidator(Validator[Any, dict[str, Any]]):
    """Reads a person's answer to one waiting call into the node's outputs. Built for each answer, with
    the questions the call's ``Ask`` asked."""                                                 # bad

class AnswerValidator(Validator[Any, dict[str, Any]]):
    """A validator for a person's answer to a node that raised ``Ask``.

    It checks the answer and turns it into the node's outputs.

    Raises (when called):
        InvalidCallError: One ``invalid_answer`` problem per fault.
    """                                                                                        # good
```

## No history in code

Write code, comments, docstrings and docs about what the code does. Leave out "renamed from", "formerly", "no longer", dates, plan files, decision ids and PR numbers. History goes in git, and sparingly in `CHANGELOG.md`.

Why: a reader wants to know what the code does. How it got there is in git.

```toml
# The order of the module map (plans: D-L143), with its third row split.        # bad
# Packages on one line with " | " between them import none of each other.       # good
```
