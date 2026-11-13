"""Every kind of problem conductor's core reports, one function each, in ``PROBLEMS``.

The kinds are declared in three modules, one per section of the problems page, each with the catalogue of its
section: ``compile_problems.py`` (``COMPILE_PROBLEMS``, in ``graph.problems``), ``call_problems.py``
(``CALL_PROBLEMS``, raised as ``InvalidCallError``) and ``run_problems.py`` (``RUN_PROBLEMS``, why a run ended
early). ``PROBLEMS`` joins the three, in the page's order. A module that reports a kind imports the module that
declares it and calls the kind: ``compile_problems.cycle(node_id)``.

A kind is a function under ``@COMPILE_PROBLEMS.problem()``, or the catalogue of its section. Its name is the problem's code. Its positional-only parameters
(before the ``/``) and its ``*loc`` are the parts of ``loc``, and a named one is a key of ``ctx`` too; its keyword-only
parameters are the other keys of ``ctx``.
It returns the message. Calling it makes the problem::

    unknown_input("upper", "txet", inputs=["text"])
    # Problem(type="unknown_input", loc=("upper", "txet"), msg="'upper' has no input 'txet'; it has text.",
    #         ctx={"node_id": "upper", "field": "txet", "inputs": ["text"]})

A type checker checks every call against the function's signature; ``ProblemCatalogue.problem`` in ``type.py`` gives
the rules for the signature in full. A few kinds name their code in the decorator,
``@COMPILE_PROBLEMS.problem("type_mismatch", variant="make_a_series")``: a second message under that code, for one particular
mistake, and the problem's ``ctx.variant`` says which.

Each kind's docstring is its entry on the problems page (``python -m conductor.about problems``): what is wrong and
how to fix it, a line for each ``loc`` and ``ctx`` parameter, and an example. The example runs after the setup at
the top of that page (``conductor/about/problems.md``), which makes a ``registry`` of small nodes (``upper``,
``greet``, ``count``, ``words``, ``join``, ``route``, ``approve``, ``who``, ``pick``, ``contacts``) and a helper for
each section: ``problems(record)`` returns the problems of a graph record compiled with ``registry`` or the registry
it is given, ``refused(call)`` the problems a refused call raises, and ``why_it_stopped(run)`` the problem a run
ended on. Its last line calls its section's helper, and a test checks it gives problems of this kind and no other.
"""

from __future__ import annotations

from .call_problems import CALL_PROBLEMS
from .compile_problems import COMPILE_PROBLEMS
from .run_problems import RUN_PROBLEMS
from .type import ProblemCatalogue

#: Every kind of problem conductor's core reports.
PROBLEMS = ProblemCatalogue.joined(COMPILE_PROBLEMS, CALL_PROBLEMS, RUN_PROBLEMS)
