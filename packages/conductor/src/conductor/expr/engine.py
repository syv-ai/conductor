"""Sandboxed CEL-compatible expression evaluator.

This is a self-contained implementation of a useful CEL subset — designed to
be embedded in Conductor without pulling in an external parser generator.

Grammar (informal):

    expr      := ternary
    ternary   := or ('?' expr ':' expr)?
    or        := and ('||' and)*
    and       := equality ('&&' equality)*
    equality  := comparison (('==' | '!=') comparison)*
    comparison:= additive (('<' | '<=' | '>' | '>=' | 'in') additive)*
    additive  := multiplicative (('+' | '-') multiplicative)*
    multiplicative := unary (('*' | '/' | '%') unary)*
    unary     := ('!' | '-') unary | postfix
    postfix   := primary ('.' IDENT | '[' expr ']' | '(' args? ')')*
    primary   := NUMBER | STRING | 'true' | 'false' | 'null' | 'None'
               | IDENT | '(' expr ')' | list_lit | map_lit
    list_lit  := '[' (expr (',' expr)*)? ']'
    map_lit   := '{' (pair (',' pair)*)? '}'
    pair      := expr ':' expr

Built-in functions (callable on values or bare):

    size(x)                 — length of string/list/map
    has(x, key)             — map or dict contains key
    contains(s, sub)        — string contains substring
    startsWith(s, prefix)   — string starts with
    endsWith(s, suffix)     — string ends with
    matches(s, pattern)     — regex match (full)
    string(x) / int(x) / double(x) / bool(x) — conversions
    lower(s) / upper(s)
    exists(path)            — checks that a dotted path resolves
    int(...) and double(...) accept None by raising.

Context is a plain dict; identifier ``foo`` resolves to ``context["foo"]``.
Special convenience: ``$`` is the root context, so ``$.foo.bar`` works too.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


class ExpressionError(Exception):
    """Base class for expression parse/type/runtime failures."""


class ExpressionParseError(ExpressionError):
    """Raised when parsing fails."""


class ExpressionTypeError(ExpressionError):
    """Raised when operand types are incompatible."""


class ExpressionRuntimeError(ExpressionError):
    """Raised when expression evaluation fails at runtime (missing key, etc.)."""


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


_TOKEN_SPEC = [
    ("NUMBER", r"\d+\.\d+|\d+"),
    ("STRING", r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\""),
    ("OP", r"==|!=|<=|>=|&&|\|\||[+\-*/%<>!=?:,().\[\]{}]"),
    ("IDENT", r"\$|[A-Za-z_][A-Za-z_0-9]*"),
    ("SKIP", r"[ \t\n\r]+"),
    ("MISMATCH", r"."),
]
_TOKEN_RE = re.compile("|".join(f"(?P<{n}>{p})" for n, p in _TOKEN_SPEC))

_KEYWORDS = {"true", "false", "null", "None", "in", "and", "or", "not"}


@dataclass
class _Token:
    kind: str
    value: str
    pos: int


def _tokenize(source: str) -> list[_Token]:
    tokens: list[_Token] = []
    for m in _TOKEN_RE.finditer(source):
        kind = m.lastgroup or ""
        val = m.group()
        if kind == "SKIP":
            continue
        if kind == "MISMATCH":
            raise ExpressionParseError(
                f"Unexpected character {val!r} at position {m.start()}"
            )
        if kind == "IDENT" and val in _KEYWORDS:
            tokens.append(_Token("KW", val, m.start()))
        else:
            tokens.append(_Token(kind, val, m.start()))
    tokens.append(_Token("EOF", "", len(source)))
    return tokens


# ---------------------------------------------------------------------------
# AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Node:
    pass


@dataclass(frozen=True)
class _Literal(_Node):
    value: Any


@dataclass(frozen=True)
class _Ident(_Node):
    name: str


@dataclass(frozen=True)
class _Attr(_Node):
    target: _Node
    name: str


@dataclass(frozen=True)
class _Index(_Node):
    target: _Node
    key: _Node


@dataclass(frozen=True)
class _Call(_Node):
    target: _Node
    args: tuple[_Node, ...]


@dataclass(frozen=True)
class _Unary(_Node):
    op: str
    operand: _Node


@dataclass(frozen=True)
class _Binary(_Node):
    op: str
    left: _Node
    right: _Node


@dataclass(frozen=True)
class _Ternary(_Node):
    cond: _Node
    then: _Node
    otherwise: _Node


@dataclass(frozen=True)
class _ListLit(_Node):
    items: tuple[_Node, ...]


@dataclass(frozen=True)
class _MapLit(_Node):
    items: tuple[tuple[_Node, _Node], ...]


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[_Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self, offset: int = 0) -> _Token:
        return self.tokens[self.pos + offset]

    def advance(self) -> _Token:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def match(self, kind: str, value: str | None = None) -> bool:
        tok = self.peek()
        if tok.kind == kind and (value is None or tok.value == value):
            self.advance()
            return True
        return False

    def expect(self, kind: str, value: str | None = None) -> _Token:
        tok = self.peek()
        if tok.kind == kind and (value is None or tok.value == value):
            return self.advance()
        expected = f"{kind}({value!r})" if value else kind
        raise ExpressionParseError(
            f"Expected {expected} at position {tok.pos}, got {tok.kind}({tok.value!r})"
        )

    # ---- grammar

    def parse(self) -> _Node:
        node = self.parse_ternary()
        if self.peek().kind != "EOF":
            tok = self.peek()
            raise ExpressionParseError(
                f"Unexpected token {tok.kind}({tok.value!r}) at position {tok.pos}"
            )
        return node

    def parse_ternary(self) -> _Node:
        cond = self.parse_or()
        if self.match("OP", "?"):
            then = self.parse_ternary()
            self.expect("OP", ":")
            otherwise = self.parse_ternary()
            return _Ternary(cond, then, otherwise)
        return cond

    def parse_or(self) -> _Node:
        left = self.parse_and()
        while self.peek().kind == "OP" and self.peek().value == "||":
            self.advance()
            right = self.parse_and()
            left = _Binary("||", left, right)
        # Python-style `or`
        while self.peek().kind == "KW" and self.peek().value == "or":
            self.advance()
            right = self.parse_and()
            left = _Binary("||", left, right)
        return left

    def parse_and(self) -> _Node:
        left = self.parse_equality()
        while self.peek().kind == "OP" and self.peek().value == "&&":
            self.advance()
            right = self.parse_equality()
            left = _Binary("&&", left, right)
        while self.peek().kind == "KW" and self.peek().value == "and":
            self.advance()
            right = self.parse_equality()
            left = _Binary("&&", left, right)
        return left

    def parse_equality(self) -> _Node:
        left = self.parse_comparison()
        while self.peek().kind == "OP" and self.peek().value in ("==", "!="):
            op = self.advance().value
            right = self.parse_comparison()
            left = _Binary(op, left, right)
        return left

    def parse_comparison(self) -> _Node:
        left = self.parse_additive()
        while True:
            tok = self.peek()
            if tok.kind == "OP" and tok.value in ("<", "<=", ">", ">="):
                op = self.advance().value
                right = self.parse_additive()
                left = _Binary(op, left, right)
            elif tok.kind == "KW" and tok.value == "in":
                self.advance()
                right = self.parse_additive()
                left = _Binary("in", left, right)
            else:
                break
        return left

    def parse_additive(self) -> _Node:
        left = self.parse_multiplicative()
        while self.peek().kind == "OP" and self.peek().value in ("+", "-"):
            op = self.advance().value
            right = self.parse_multiplicative()
            left = _Binary(op, left, right)
        return left

    def parse_multiplicative(self) -> _Node:
        left = self.parse_unary()
        while self.peek().kind == "OP" and self.peek().value in ("*", "/", "%"):
            op = self.advance().value
            right = self.parse_unary()
            left = _Binary(op, left, right)
        return left

    def parse_unary(self) -> _Node:
        tok = self.peek()
        if tok.kind == "OP" and tok.value in ("!", "-"):
            op = self.advance().value
            operand = self.parse_unary()
            return _Unary(op, operand)
        if tok.kind == "KW" and tok.value == "not":
            self.advance()
            operand = self.parse_unary()
            return _Unary("!", operand)
        return self.parse_postfix()

    def parse_postfix(self) -> _Node:
        node = self.parse_primary()
        while True:
            tok = self.peek()
            if tok.kind == "OP" and tok.value == ".":
                self.advance()
                name_tok = self.expect("IDENT")
                node = _Attr(node, name_tok.value)
            elif tok.kind == "OP" and tok.value == "[":
                self.advance()
                key = self.parse_ternary()
                self.expect("OP", "]")
                node = _Index(node, key)
            elif tok.kind == "OP" and tok.value == "(":
                self.advance()
                args: list[_Node] = []
                if not (self.peek().kind == "OP" and self.peek().value == ")"):
                    args.append(self.parse_ternary())
                    while self.match("OP", ","):
                        args.append(self.parse_ternary())
                self.expect("OP", ")")
                node = _Call(node, tuple(args))
            else:
                break
        return node

    def parse_primary(self) -> _Node:
        tok = self.peek()

        if tok.kind == "NUMBER":
            self.advance()
            if "." in tok.value:
                return _Literal(float(tok.value))
            return _Literal(int(tok.value))

        if tok.kind == "STRING":
            self.advance()
            return _Literal(_unescape_string(tok.value))

        if tok.kind == "KW":
            self.advance()
            if tok.value == "true":
                return _Literal(True)
            if tok.value == "false":
                return _Literal(False)
            if tok.value in ("null", "None"):
                return _Literal(None)
            raise ExpressionParseError(
                f"Unexpected keyword {tok.value!r} at position {tok.pos}"
            )

        if tok.kind == "IDENT":
            self.advance()
            return _Ident(tok.value)

        if tok.kind == "OP" and tok.value == "(":
            self.advance()
            node = self.parse_ternary()
            self.expect("OP", ")")
            return node

        if tok.kind == "OP" and tok.value == "[":
            self.advance()
            items: list[_Node] = []
            if not (self.peek().kind == "OP" and self.peek().value == "]"):
                items.append(self.parse_ternary())
                while self.match("OP", ","):
                    items.append(self.parse_ternary())
            self.expect("OP", "]")
            return _ListLit(tuple(items))

        if tok.kind == "OP" and tok.value == "{":
            self.advance()
            pairs: list[tuple[_Node, _Node]] = []
            if not (self.peek().kind == "OP" and self.peek().value == "}"):
                key = self.parse_ternary()
                self.expect("OP", ":")
                val = self.parse_ternary()
                pairs.append((key, val))
                while self.match("OP", ","):
                    key = self.parse_ternary()
                    self.expect("OP", ":")
                    val = self.parse_ternary()
                    pairs.append((key, val))
            self.expect("OP", "}")
            return _MapLit(tuple(pairs))

        raise ExpressionParseError(
            f"Unexpected token {tok.kind}({tok.value!r}) at position {tok.pos}"
        )


def _unescape_string(raw: str) -> str:
    """Strip quotes and process escape sequences (a minimal subset)."""
    body = raw[1:-1]
    out = []
    i = 0
    while i < len(body):
        c = body[i]
        if c == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            out.append({
                "n": "\n", "t": "\t", "r": "\r",
                "\\": "\\", "'": "'", '"': '"', "0": "\0",
            }.get(nxt, nxt))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


_UNDEFINED = object()


def _resolve_ident(name: str, ctx: dict[str, Any]) -> Any:
    if name == "$":
        return ctx
    if name in ctx:
        return ctx[name]
    raise ExpressionRuntimeError(f"Undefined identifier: {name!r}")


def _attr(target: Any, name: str) -> Any:
    # Sandbox: never expose arbitrary Python attributes. Only dicts and
    # explicitly-typed value shapes (pydantic models, dataclasses) can yield
    # attribute values; methods are routed through the _METHODS allowlist
    # by the caller, not reachable here.
    if target is None:
        raise ExpressionRuntimeError(f"Cannot access '.{name}' on null")
    if isinstance(target, dict):
        if name in target:
            return target[name]
        raise ExpressionRuntimeError(f"Key {name!r} missing on map")
    if isinstance(target, BaseModel):
        if name in type(target).model_fields:
            return getattr(target, name)
        raise ExpressionRuntimeError(
            f"No field {name!r} on {type(target).__name__}"
        )
    if dataclasses.is_dataclass(target) and not isinstance(target, type):
        field_names = {f.name for f in dataclasses.fields(target)}
        if name in field_names:
            return getattr(target, name)
        raise ExpressionRuntimeError(
            f"No field {name!r} on {type(target).__name__}"
        )
    raise ExpressionRuntimeError(
        f"Cannot access '.{name}' on value of type {type(target).__name__}"
    )


def _index(target: Any, key: Any) -> Any:
    if target is None:
        raise ExpressionRuntimeError("Cannot index into null")
    try:
        return target[key]
    except (KeyError, IndexError, TypeError) as e:
        raise ExpressionRuntimeError(f"Index error: {e}") from e


# Caps for the `matches` builtin/method to limit ReDoS blast radius.
# A crafted pattern like `(a+)+$` against a moderately long string can hang
# the scheduler thread; we can't cheaply bound regex execution time in the
# stdlib, so we bound the inputs instead. Hosts that need unbounded regex
# should pre-validate or swap in re2.
_MAX_REGEX_PATTERN_LEN = 256
_MAX_REGEX_INPUT_LEN = 64 * 1024


def _safe_fullmatch(pattern: str, text: str) -> bool:
    if not isinstance(pattern, str):
        raise ExpressionTypeError(
            f"matches() pattern must be string, got {type(pattern).__name__}"
        )
    if not isinstance(text, str):
        raise ExpressionTypeError(
            f"matches() target must be string, got {type(text).__name__}"
        )
    if len(pattern) > _MAX_REGEX_PATTERN_LEN:
        raise ExpressionRuntimeError(
            f"matches() pattern exceeds {_MAX_REGEX_PATTERN_LEN} chars"
        )
    if len(text) > _MAX_REGEX_INPUT_LEN:
        raise ExpressionRuntimeError(
            f"matches() input exceeds {_MAX_REGEX_INPUT_LEN} chars"
        )
    try:
        return re.fullmatch(pattern, text) is not None
    except re.error as e:
        raise ExpressionRuntimeError(f"Invalid regex {pattern!r}: {e}") from e


_BUILTIN_FUNCS = {
    "size": lambda x: len(x),
    "has": lambda m, k: (k in m) if isinstance(m, dict) else (k in m),
    "contains": lambda s, sub: sub in s,
    "startsWith": lambda s, p: s.startswith(p),
    "endsWith": lambda s, p: s.endswith(p),
    "matches": _safe_fullmatch,
    "lower": lambda s: s.lower(),
    "upper": lambda s: s.upper(),
    "string": lambda x: str(x),
    "int": lambda x: int(x),
    "double": lambda x: float(x),
    "bool": lambda x: bool(x),
    "exists": lambda x: x is not None,
    "min": min,
    "max": max,
    "abs": abs,
}


_METHODS = {
    "contains": lambda self, sub: sub in self,
    "startsWith": lambda self, p: self.startswith(p),
    "endsWith": lambda self, p: self.endswith(p),
    "matches": lambda self, p: _safe_fullmatch(p, self),
    "lower": lambda self: self.lower() if isinstance(self, str) else self,
    "upper": lambda self: self.upper() if isinstance(self, str) else self,
    "size": lambda self: len(self),
    "keys": lambda self: list(self.keys()) if isinstance(self, dict) else [],
    "values": lambda self: list(self.values()) if isinstance(self, dict) else [],
}


def _eval(node: _Node, ctx: dict[str, Any]) -> Any:
    if isinstance(node, _Literal):
        return node.value
    if isinstance(node, _Ident):
        return _resolve_ident(node.name, ctx)
    if isinstance(node, _Attr):
        # Special: `foo.method()` — detect via _Call below; bare attr is member access.
        target = _eval(node.target, ctx)
        try:
            return _attr(target, node.name)
        except ExpressionRuntimeError:
            if node.name in _METHODS:
                return _MethodRef(target, node.name)
            raise
    if isinstance(node, _Index):
        target = _eval(node.target, ctx)
        key = _eval(node.key, ctx)
        return _index(target, key)
    if isinstance(node, _Call):
        return _eval_call(node, ctx)
    if isinstance(node, _Unary):
        v = _eval(node.operand, ctx)
        if node.op == "!":
            return not bool(v)
        if node.op == "-":
            if not isinstance(v, (int, float)):
                raise ExpressionTypeError(f"Cannot negate {type(v).__name__}")
            return -v
        raise ExpressionRuntimeError(f"Unknown unary op {node.op!r}")
    if isinstance(node, _Binary):
        return _eval_binary(node, ctx)
    if isinstance(node, _Ternary):
        c = _eval(node.cond, ctx)
        if bool(c):
            return _eval(node.then, ctx)
        return _eval(node.otherwise, ctx)
    if isinstance(node, _ListLit):
        return [_eval(item, ctx) for item in node.items]
    if isinstance(node, _MapLit):
        return {_eval(k, ctx): _eval(v, ctx) for k, v in node.items}
    raise ExpressionRuntimeError(f"Unknown node type {type(node).__name__}")


@dataclass(frozen=True)
class _MethodRef:
    target: Any
    name: str


def _eval_call(node: _Call, ctx: dict[str, Any]) -> Any:
    # Bare identifier call: size(x), has(m, k), …
    if isinstance(node.target, _Ident) and node.target.name in _BUILTIN_FUNCS:
        args = [_eval(a, ctx) for a in node.args]
        try:
            return _BUILTIN_FUNCS[node.target.name](*args)
        except ExpressionError:
            raise
        except Exception as e:
            raise ExpressionRuntimeError(
                f"Built-in '{node.target.name}' failed: {e}"
            ) from e

    # Method-style call: s.contains("x"), m.keys(), …
    target_val = _eval(node.target, ctx)
    if isinstance(target_val, _MethodRef):
        fn = _METHODS.get(target_val.name)
        if fn is None:
            raise ExpressionRuntimeError(f"No method {target_val.name!r}")
        try:
            return fn(target_val.target, *[_eval(a, ctx) for a in node.args])
        except Exception as e:
            raise ExpressionRuntimeError(
                f"Method '{target_val.name}' failed: {e}"
            ) from e

    # No arbitrary-callable fallback: exposing Python callables would let a
    # flow author invoke any method on any object placed in the expression
    # context. All legitimate calls go through _BUILTIN_FUNCS or _METHODS.
    raise ExpressionRuntimeError(f"Value {target_val!r} is not callable")


def _eval_binary(node: _Binary, ctx: dict[str, Any]) -> Any:
    # Short-circuit logical operators
    if node.op == "&&":
        left = _eval(node.left, ctx)
        if not bool(left):
            return False
        return bool(_eval(node.right, ctx))
    if node.op == "||":
        left = _eval(node.left, ctx)
        if bool(left):
            return True
        return bool(_eval(node.right, ctx))

    left = _eval(node.left, ctx)
    right = _eval(node.right, ctx)

    if node.op == "==":
        return left == right
    if node.op == "!=":
        return left != right
    if node.op == "<":
        return left < right
    if node.op == "<=":
        return left <= right
    if node.op == ">":
        return left > right
    if node.op == ">=":
        return left >= right
    if node.op == "in":
        return left in right
    if node.op == "+":
        if isinstance(left, str) or isinstance(right, str):
            return f"{left}{right}" if isinstance(left, str) and isinstance(right, str) \
                else _plus_mixed(left, right)
        if isinstance(left, list) and isinstance(right, list):
            return left + right
        return left + right
    if node.op == "-":
        return left - right
    if node.op == "*":
        return left * right
    if node.op == "/":
        if right == 0:
            raise ExpressionRuntimeError("Division by zero")
        # CEL: int/int yields int, truncated toward zero (not floored).
        # Python's `//` floors, which differs for mixed-sign operands:
        # CEL says `-5 / 2 == -2` but `-5 // 2 == -3`. Use divmod + adjust.
        if isinstance(left, int) and isinstance(right, int) \
                and not isinstance(left, bool) and not isinstance(right, bool):
            q, r = divmod(left, right)
            if r != 0 and (left < 0) != (right < 0):
                q += 1
            return q
        return left / right
    if node.op == "%":
        return left % right
    raise ExpressionRuntimeError(f"Unknown binary op {node.op!r}")


def _plus_mixed(left: Any, right: Any) -> Any:
    if isinstance(left, str):
        return left + str(right)
    return str(left) + right


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Expression:
    """A parsed, reusable expression. Call ``evaluate(ctx)`` to run it."""

    source: str
    _ast: _Node

    def evaluate(self, ctx: dict[str, Any]) -> Any:
        """Run the expression against a context dict. Returns any value."""
        return _eval(self._ast, ctx)

    def evaluate_bool(self, ctx: dict[str, Any]) -> bool:
        """Evaluate and coerce to a Python bool. Type-checks CEL semantics."""
        result = self.evaluate(ctx)
        return bool(result)

    def identifiers(self) -> set[str]:
        """Return the set of top-level identifiers referenced by this expression."""
        seen: set[str] = set()
        _collect_idents(self._ast, seen)
        return seen


def _collect_idents(node: _Node, out: set[str]) -> None:
    if isinstance(node, _Ident):
        out.add(node.name)
        return
    if isinstance(node, _Attr):
        _collect_idents(node.target, out)
        return
    if isinstance(node, _Index):
        _collect_idents(node.target, out)
        _collect_idents(node.key, out)
        return
    if isinstance(node, _Call):
        # Don't collect built-in function names as identifiers.
        if isinstance(node.target, _Ident) and node.target.name in _BUILTIN_FUNCS:
            pass
        else:
            _collect_idents(node.target, out)
        for arg in node.args:
            _collect_idents(arg, out)
        return
    if isinstance(node, _Unary):
        _collect_idents(node.operand, out)
        return
    if isinstance(node, _Binary):
        _collect_idents(node.left, out)
        _collect_idents(node.right, out)
        return
    if isinstance(node, _Ternary):
        _collect_idents(node.cond, out)
        _collect_idents(node.then, out)
        _collect_idents(node.otherwise, out)
        return
    if isinstance(node, _ListLit):
        for item in node.items:
            _collect_idents(item, out)
        return
    if isinstance(node, _MapLit):
        for k, v in node.items:
            _collect_idents(k, out)
            _collect_idents(v, out)
        return


def parse(source: str) -> Expression:
    """Parse a CEL expression. Raises :class:`ExpressionParseError` on bad syntax."""
    if not isinstance(source, str) or not source.strip():
        raise ExpressionParseError("Expression is empty")
    try:
        tokens = _tokenize(source)
        ast = _Parser(tokens).parse()
    except ExpressionError:
        raise
    except Exception as e:  # defensive
        raise ExpressionParseError(f"Parse error: {e}") from e
    return Expression(source=source, _ast=ast)


def evaluate(source: str, ctx: dict[str, Any]) -> Any:
    """Parse and evaluate in one call."""
    return parse(source).evaluate(ctx)
