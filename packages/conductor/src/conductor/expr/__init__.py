"""Expression language module — CEL-like mini-evaluator.

Conductor uses a small CEL-compatible expression language for:

* Edge ``when`` guards on decision nodes
* ``while-start`` loop conditions
* ``idempotency_key`` expressions on nodes
* Trigger input-mapping and correlation expressions
* Subprocess input/output mapping

The evaluator implements a safe, sandboxed subset of the Common Expression
Language (CEL) spec — literals, identifiers (with dotted and index access),
the usual arithmetic/comparison/logical operators, string concatenation,
and a small library of built-in functions. It has no I/O, no side effects,
and never calls arbitrary Python.

The public surface is:

* :class:`Expression` — a parsed, reusable expression.
* :func:`evaluate` — one-shot parse+eval helper.
* :class:`ExpressionError` — raised for parse or eval failures.
"""

from conductor.expr.engine import (
    Expression,
    ExpressionError,
    ExpressionParseError,
    ExpressionRuntimeError,
    ExpressionTypeError,
    evaluate,
    parse,
)

__all__ = [
    "Expression",
    "ExpressionError",
    "ExpressionParseError",
    "ExpressionRuntimeError",
    "ExpressionTypeError",
    "evaluate",
    "parse",
]
