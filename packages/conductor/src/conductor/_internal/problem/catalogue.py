"""Every kind of problem conductor's core reports, one function each, in ``PROBLEMS``.

The kinds are declared in three modules, one per section of the problems page, and each module starts its section,
so a kind is in the section of the module it is declared in: ``compiling.py`` (in ``graph.problems``),
``refused.py`` (raised as ``InvalidCallError``) and ``stopped.py`` (why a run ended early). A module that reports
one reads it here, ``from conductor._internal.problem import catalogue as problem``, and calls ``problem.cycle(node_id)``.

A kind is a function under ``@PROBLEMS.problem()``. Its name is the problem's code. Its positional-only parameters
(before the ``/``) and its ``*loc`` are the parts of ``loc``, and a named one is a key of ``ctx`` too; its keyword-only
parameters are the other keys of ``ctx``.
It returns the message. Calling it makes the problem::

    unknown_input("upper", "txet", inputs=["text"])
    # Problem(type="unknown_input", loc=("upper", "txet"), msg="'upper' has no input 'txet'; it has text.",
    #         ctx={"node_id": "upper", "field": "txet", "inputs": ["text"]})

A type checker checks every call against the function's signature; ``ProblemCatalogue.problem`` in ``type.py`` gives
the rules for the signature in full. A few kinds name their code in the decorator,
``@PROBLEMS.problem("type_mismatch", variant="make_a_series")``: a second message under that code, for one particular
mistake, and the problem's ``ctx.variant`` says which. ``of_a_hook=True`` marks a problem in a node's ``compute_...``
hook, which the node's developer fixes rather than the graph's author; ``check_registry`` reports those as its own.

Each kind's docstring is its entry on the problems page (``python -m conductor.about problems``): what is wrong and
how to fix it, a line for each ``loc`` and ``ctx`` parameter, and an example. The example runs after the setup at
the top of that page (``conductor/about/problems.md``), which makes a ``registry`` of small nodes (``upper``,
``greet``, ``count``, ``words``, ``join``, ``route``, ``approve``, ``who``, ``pick``, ``contacts``) and a helper for
each section: ``problems(record)`` returns the problems of a graph record compiled with ``registry`` or the registry
it is given, ``refused(call)`` the problems a refused call raises, and ``why_it_stopped(run)`` the problem a run
ended on. Its last line calls its section's helper, and a test checks it gives problems of this kind and no other.
"""

from __future__ import annotations

from .type import ProblemCatalogue

#: Every kind of problem conductor's core reports.
PROBLEMS: ProblemCatalogue = ProblemCatalogue()

# The sections, in the page's order. Each declares its kinds on PROBLEMS, so it is imported after PROBLEMS is made.
from .compiling import *  # noqa: E402, F403
from .refused import *  # noqa: E402, F403
from .stopped import *  # noqa: E402, F403
