"""Tests for the messages of conductor's problems, and for the problems page.

Every kind of problem conductor can report has a message, and an entry with an example on the problems page
(``python -m conductor.about problems``). These tests check that each message says how to fix the problem, and that
each example on the page really makes the problem it sits under.
"""

from __future__ import annotations

import ast
import linecache
import re
import sys
from ast import Call, Expr, Expression, Module, Name
from pathlib import Path
from types import ModuleType

import pytest
from conductor import DECLARATION_PROBLEMS, PROBLEMS, Problem, ProblemKind
from conductor.about import problems_page
from conductor.testing import REGISTRY_CHECK_PROBLEMS
from pytest import MonkeyPatch

_ROOT = Path(__file__).parents[2]
_PACKAGE = _ROOT / "packages" / "conductor" / "src" / "conductor"
_TEMPLATE = (_PACKAGE / "about" / "problems.md").read_text()
#: The kinds whose message has no fix of its own. A node's failure uses the node's own words. An outside failure
#: has nothing a person can act on. And an answer's ``reason`` is pydantic's account of what is wrong with it.
_NO_FIX = {"node_error", "external_failed", "invalid_answer"}
#: The kinds whose message is the one the node gave its ``NodeError``. So an example's message isn't the kind's own.
_MESSAGE_IS_THE_NODES = {"node_error"}


def test_every_message_says_what_to_do() -> None:
    """Each message says what is wrong, and then how to fix it.

    The test looks for a second clause after a full stop, colon or semicolon. The fix may be a list of the values to
    pick from. The kinds in ``_NO_FIX`` are left out.
    """
    sentences = [(kind.code, kind.sentence) for kind in [*PROBLEMS, *DECLARATION_PROBLEMS]]
    without = [
        code
        for code, text in sentences
        if code not in _NO_FIX and not re.search(r"[;:.] \S", text.removesuffix(": {reason}"))
    ]
    assert without == []


def _setup() -> str:
    """Returns the setup at the top of the problems page, which every example builds on: its first Python block."""
    return re.findall(r"```python\n(.*?)```", _TEMPLATE, re.S)[0]


def _is_of_kind(kind: ProblemKind[...], problem: Problem) -> bool:
    """Whether ``problem`` is a problem of ``kind``.

    Its code must be the kind's. Its message must be one the kind's sentence could write, with any text
    in place of each ``{field}``.
    """
    if problem.type != kind.code:
        return False
    if kind.code in _MESSAGE_IS_THE_NODES:
        return True
    pattern = re.escape(kind.sentence)
    for replacement_field in re.findall(r"\\\{\w+\\\}", pattern):
        pattern = pattern.replace(replacement_field, ".*")
    return re.fullmatch(pattern, problem.msg, re.S) is not None


def _examples(page: str) -> dict[str, str]:
    """Returns the example of each kind of problem on the printed page, by the kind's code.

    A kind's example is the first Python block in its section. A kind with no example gets an empty string.
    """
    examples = {}
    for code, body in re.findall(r"^### `(\w+)`\n(.*?)(?=^##|\Z)", page, re.S | re.M):
        block = re.search(r"```python\n(.*?)```", body, re.S)
        examples[code] = block[1] if block else ""
    return examples


_PAGE_EXAMPLES = _examples(problems_page())

#: Every kind with an example on the problems page.
_KINDS = [*PROBLEMS, *DECLARATION_PROBLEMS, *REGISTRY_CHECK_PROBLEMS]


@pytest.mark.parametrize("kind", _KINDS, ids=[kind.code for kind in _KINDS])
def test_a_kinds_example_makes_its_problem_and_no_other(
    kind: ProblemKind[...], monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    """Each example on the page makes problems of its own kind, and of no other.

    The example runs on its own, after the page's setup. Its last line gives the problems it made.
    """
    monkeypatch.chdir(tmp_path)
    page = ModuleType("problems_page")  # a dataclass in an example looks up its module by name
    monkeypatch.setitem(sys.modules, page.__name__, page)
    example = _PAGE_EXAMPLES[kind.code]
    assert example, f"{kind.code} has no example"
    *body, last = ast.parse(example).body
    assert isinstance(last, Expr), f"{kind.code}: the example's last line gives its problems"
    exec(compile(_setup(), "problems page: setup", "exec"), page.__dict__)  # noqa: S102  the page's own code
    # The example's source can be read, as a file's or a notebook cell's can, so @node reads it as it reads theirs.
    where = f"problems page: {kind.code}"
    monkeypatch.setitem(
        linecache.cache, where, (len(example), None, example.splitlines(keepends=True), where)
    )
    exec(compile(Module(body=body, type_ignores=[]), where, "exec"), page.__dict__)  # noqa: S102
    made = eval(compile(Expression(last.value), where, "eval"), page.__dict__)  # noqa: S307
    made = made if isinstance(made, list) else [made]
    assert made and [problem for problem in made if not _is_of_kind(kind, problem)] == []


#: The helper in the page's setup that each section's examples get their problems with. Each section says where a
#: host finds its problems: in ``graph.problems``, in a refused call's ``InvalidCallError``, as a run's cause, or in
#: a ``NodeDeclarationError``.
_MET_BY = {
    "Compiling a graph": "problems",
    "Refused calls": "refused",
    "Why a run stopped": "why_it_stopped",
    "Declaring a node": "declaration_refused",
}
_SECTIONED = [*PROBLEMS, *DECLARATION_PROBLEMS]


@pytest.mark.parametrize("kind", _SECTIONED, ids=[kind.code for kind in _SECTIONED])
def test_a_kinds_example_meets_its_problem_where_its_sections_text_says(
    kind: ProblemKind[...],
) -> None:
    """Each example gets its problem where its section of the page says a host finds it.

    The example's last line calls its section's helper. The test above runs that line, so the problem really turns up
    there.
    """
    *_, last = ast.parse(_PAGE_EXAMPLES[kind.code]).body
    assert (
        isinstance(last, Expr)
        and isinstance(last.value, Call)
        and isinstance(last.value.func, Name)
    )
    assert last.value.func.id == _MET_BY[kind.section]


def test_every_node_of_the_setup_runs(monkeypatch: MonkeyPatch) -> None:
    """Every node in the page's setup runs without a problem.

    Most examples only compile graphs of these nodes and never call them. So ``check_registry`` calls each one here.
    """
    page = ModuleType("problems_page")
    monkeypatch.setitem(sys.modules, page.__name__, page)
    exec(compile(_setup(), "problems page: setup", "exec"), page.__dict__)  # noqa: S102
    examples = {
        "upper": [{"text": "a"}],
        "greet": [{"name": "Ida", "greeting": "Hi"}],
        "count": [{"n": 2}],
        "words": [{"text": "a b"}],
        "join": [{"texts": ["a", "b"]}],
        "route": [{"text": "ab"}, {"text": "abcdef"}],
        "approve": [{"amount": 5}],
        "who": [{}],
        "pick": [{"names": "a,b"}],
        "contacts": [{"text": "Ida ida@example.com"}],
    }
    assert set(examples) == {definition.id for definition in page.registry}
    assert (
        page.check_registry(
            page.registry, examples=examples, context=page.HostContext(page.Caller("Ida"))
        )
        == []
    )
