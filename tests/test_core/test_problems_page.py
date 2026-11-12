"""Every message conductor writes says what to do, and every example on the problems page makes the problem it is listed under."""

from __future__ import annotations

import ast
import re
import sys
from ast import Expr, Expression, Module
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from conductor import DECLARATION_PROBLEMS, PROBLEMS, Problem, ProblemKind
from conductor.about import _kind_docs
from conductor.testing.problems import REGISTRY_CHECK_PROBLEMS
from conductor_providers.fastapi.testing.problems import RUN_STORE_CHECK_PROBLEMS
from pytest import MonkeyPatch

_ROOT = Path(__file__).parents[2]
_PACKAGE = _ROOT / "packages" / "conductor" / "src" / "conductor"
_TEMPLATE = (_PACKAGE / "about" / "problems.md").read_text()
_SERVING_RUNS = (_ROOT / "docs" / "serving-runs.md").read_text()
#: Messages that carry no fix of their own: a node's failure keeps the node's words, an outside failure has none a
#: person can act on, and an answer's ``reason`` is pydantic's account of what the answer lacks.
_NO_FIX = {"failed", "external_failed", "invalid_answer"}
#: The kinds whose message is the one the node gave its ``NodeError``, so an example's message is not the kind's own.
_MESSAGE_IS_THE_NODES = {"failed"}


def test_every_message_says_what_to_do() -> None:
    """A message carries its fix: a clause after what is wrong, which may be the alternatives to pick from."""
    sentences = [(kind.code, kind.sentence) for kind in [*PROBLEMS, *DECLARATION_PROBLEMS]]
    without = [code for code, text in sentences if code not in _NO_FIX and not re.search(r"[;:.] \S", text.removesuffix(": {reason}"))]
    assert without == []


def _setup() -> str:
    """The setup at the top of the problems page: its first Python block."""
    return re.findall(r"```python\n(.*?)```", _TEMPLATE, re.S)[0]


def _store_setup() -> str:
    """The setup of ``check_run_store``'s examples: the Python blocks of the section on writing a store, on the page
    on serving runs."""
    section = _SERVING_RUNS[_SERVING_RUNS.index("## A store"):_SERVING_RUNS.index("## The app")]
    return "\n".join(re.findall(r"```python\n(.*?)```", section, re.S))


def _made_by(kind: ProblemKind[...], problem: Problem) -> bool:
    """Whether ``problem`` is of ``kind``: its code and variant, and a message the kind's sentence writes."""
    if (problem.type, problem.ctx.get("variant")) != (kind.code, kind.variant):
        return False
    if kind.code in _MESSAGE_IS_THE_NODES:
        return True
    pattern = re.escape(kind.sentence)
    for slot in re.findall(r"\\\{\w+\\\}", pattern):
        pattern = pattern.replace(slot, ".*")
    return re.fullmatch(pattern, problem.msg, re.S) is not None


#: Each kind with an example, and the setup its example runs after: that of the page its catalogue is written on.
_KINDS = [*((kind, _setup) for kind in [*PROBLEMS, *DECLARATION_PROBLEMS, *REGISTRY_CHECK_PROBLEMS]), *((kind, _store_setup) for kind in RUN_STORE_CHECK_PROBLEMS)]


@pytest.mark.parametrize(("kind", "setup"), _KINDS, ids=[f"{kind.code}-{kind._message.__name__}" for kind, _ in _KINDS])
def test_a_kinds_example_makes_its_problem_and_no_other(kind: ProblemKind[...], setup: Callable[[], str], monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """The example runs after the page's setup, alone, and its last line gives problems of this kind only."""
    monkeypatch.chdir(tmp_path)
    page = ModuleType("problems_page")  # a dataclass in an example looks its module up by name
    monkeypatch.setitem(sys.modules, page.__name__, page)
    example = _kind_docs(kind).example
    assert example, f"{kind.code} has no example"
    *body, last = ast.parse(example).body
    assert isinstance(last, Expr), f"{kind.code}: the example's last line gives its problems"
    exec(compile(setup(), "problems page: setup", "exec"), page.__dict__)  # noqa: S102  the page's own code
    exec(compile(Module(body=body, type_ignores=[]), f"problems page: {kind.code}", "exec"), page.__dict__)  # noqa: S102
    made = eval(compile(Expression(last.value), f"problems page: {kind.code}", "eval"), page.__dict__)  # noqa: S307
    made = made if isinstance(made, list) else [made]
    assert made and [problem for problem in made if not _made_by(kind, problem)] == []


def test_every_node_of_the_setup_runs(monkeypatch: MonkeyPatch) -> None:
    """The examples compile graphs of the setup's nodes, and most never call them; ``check_registry`` calls each."""
    page = ModuleType("problems_page")
    monkeypatch.setitem(sys.modules, page.__name__, page)
    exec(compile(_setup(), "problems page: setup", "exec"), page.__dict__)  # noqa: S102
    examples = {
        "upper": [{"text": "a"}], "greet": [{"name": "Ida", "greeting": "Hi"}], "count": [{"n": 2}], "words": [{"text": "a b"}],
        "join": [{"texts": ["a", "b"]}], "route": [{"text": "ab"}, {"text": "abcdef"}], "approve": [{"amount": 5}], "who": [{}],
        "pick": [{"names": "a,b"}], "contacts": [{"text": "Ida ida@example.com"}],
    }
    assert set(examples) == {definition.id for definition in page.registry}
    assert page.check_registry(page.registry, examples=examples, context=page.HostContext(page.Caller("Ida"))) == []
