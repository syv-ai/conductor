"""Every message conductor writes says what to do, and every example on the problems page makes the problem it is listed under."""

from __future__ import annotations

import ast
import contextlib
import re
import sys
from ast import Call, Constant, Expr, Expression, JoinedStr, Module, Name
from io import StringIO
from pathlib import Path
from types import ModuleType

import pytest
from conductor import PROBLEMS, Problem, ProblemKind
from conductor.about import _kind_docs
from conductor.testing.problems import REGISTRY_CHECK_PROBLEMS
from pytest import MonkeyPatch

_ROOT = Path(__file__).parents[2]
_PACKAGE = _ROOT / "packages" / "conductor" / "src" / "conductor"
_TEMPLATE = (_PACKAGE / "about" / "problems.md").read_text()
#: Messages that carry no fix of their own: a node's failure keeps the node's words, an outside failure has none a
#: person can act on, and an answer's ``reason`` is pydantic's account of what the answer lacks.
_NO_FIX = {"failed", "external_failed", "invalid_answer"}
#: The kinds whose message is the one the node gave its ``NodeError``, so an example's message is not the kind's own.
_MESSAGE_IS_THE_NODES = {"failed"}


def _template(message: ast.expr) -> str:
    """An f-string as its text with ``{slot}`` for each value, the ``{where}`` prefix naming the node left out."""
    if isinstance(message, JoinedStr):
        text = "".join(str(p.value) if isinstance(p, Constant) else "{" + ast.unparse(p.value) + "}" for p in message.values)  # type: ignore[attr-defined]
        return re.sub(r"^\{where\}", "", text)
    return str(ast.literal_eval(message))


def _declaration_messages() -> dict[str, list[str]]:
    """The messages the package raises each declaration code with; an error raised again under the code it had adds none."""
    found: dict[str, list[str]] = {}
    for path in sorted(_PACKAGE.rglob("*.py")):
        for call in ast.walk(ast.parse(path.read_text())):
            if isinstance(call, Call) and isinstance(call.func, Name) and call.func.id == "NodeDeclarationError":
                code = next(k.value for k in call.keywords if k.arg == "code")
                if isinstance(code, Constant):
                    found.setdefault(code.value, []).append(_template(call.args[0]))
    return found


def test_every_message_says_what_to_do() -> None:
    """A message carries its fix: a clause after what is wrong, which may be the alternatives to pick from."""
    sentences = [(kind.code, kind.sentence) for kind in PROBLEMS]
    without = [code for code, text in sentences if code not in _NO_FIX and not re.search(r"[;:.] \S", text.removesuffix(": {reason}"))]
    assert without == []


def test_every_declaration_message_says_what_to_do() -> None:
    """A declaration's message carries its fix too: a clause after what is wrong, past the name of the field it is about."""
    without = [
        (code, text) for code, texts in _declaration_messages().items() for text in texts
        if not re.search(r"[;,] \S", re.sub(r"^\{[^}]+\}: ", "", text))
    ]
    assert without == []


def _setup() -> str:
    """The setup at the top of the problems page: its first Python block."""
    return re.findall(r"```python\n(.*?)```", _TEMPLATE, re.S)[0]


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


_KINDS = [*PROBLEMS, *REGISTRY_CHECK_PROBLEMS]


@pytest.mark.parametrize("kind", _KINDS, ids=[f"{kind.code}-{kind._message.__name__}" for kind in _KINDS])
def test_a_kinds_example_makes_its_problem_and_no_other(kind: ProblemKind[...], monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """The example runs after the page's setup, alone, and its last line gives problems of this kind only."""
    monkeypatch.chdir(tmp_path)
    page = ModuleType("problems_page")  # a dataclass in an example looks its module up by name
    monkeypatch.setitem(sys.modules, page.__name__, page)
    example = _kind_docs(kind).example
    assert example, f"{kind.code} has no example"
    *body, last = ast.parse(example).body
    assert isinstance(last, Expr), f"{kind.code}: the example's last line gives its problems"
    exec(compile(_setup(), "problems page: setup", "exec"), page.__dict__)  # noqa: S102  the page's own code
    exec(compile(Module(body=body, type_ignores=[]), f"problems page: {kind.code}", "exec"), page.__dict__)  # noqa: S102
    made = eval(compile(Expression(last.value), f"problems page: {kind.code}", "eval"), page.__dict__)  # noqa: S307
    made = made if isinstance(made, list) else [made]
    assert made and [problem for problem in made if not _made_by(kind, problem)] == []


def test_each_declaration_example_prints_its_code(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    """The page's section on declaring a node is written by hand: each example there prints the code it is under."""
    monkeypatch.chdir(tmp_path)
    page = ModuleType("problems_page")
    monkeypatch.setitem(sys.modules, page.__name__, page)
    declaring = _TEMPLATE[_TEMPLATE.index("## Declaring a node"):_TEMPLATE.index("## `check_registry`")]
    blocks = [(match.start(), match.group(1)) for match in re.finditer(r"```python\n(.*?)```", declaring, re.S)]
    starts = [(match.start(), match.group(1)) for match in re.finditer(r"^### `([a-z_]+)`", declaring, re.M)]
    exec(compile(_setup(), "problems page: setup", "exec"), page.__dict__)  # noqa: S102
    wrong = []
    for at, block in blocks:
        code = [name for start, name in starts if start < at][-1]
        printed = StringIO()
        with contextlib.redirect_stdout(printed):
            exec(compile(block, f"problems page: {code}", "exec"), page.__dict__)  # noqa: S102
        if code not in printed.getvalue():
            wrong.append((code, printed.getvalue()))
    assert wrong == []


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
    caller = page.Caller("Ida")
    assert set(examples) == {definition.id for definition in page.registry}
    assert page.check_registry(page.registry, examples=examples, context={type(caller): caller}) == []
