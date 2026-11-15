"""``PROBLEMS``, every kind of problem conductor's core reports.

The kinds are declared in three modules, one for each section of the problems page:

- ``compile_problems`` (``COMPILE_PROBLEMS``): faults in a graph, listed in ``graph.problems``.
- ``call_problems`` (``CALL_PROBLEMS``): a call that conductor refuses, raised as ``InvalidCallError``.
- ``run_problems`` (``RUN_PROBLEMS``): why a run stopped, in ``run.cause``.

``PROBLEMS`` joins the three, in the page's order. ``ProblemCatalogue.problem`` says how a kind is declared.

Each kind's docstring is its entry on the problems page (``python -m conductor.about problems``), so it is written
for a host. It says what is wrong and how to fix it. Under ``Args`` it has a line for ``loc`` and one for each
``ctx`` key. Under ``Example`` it has code that makes the problem.

The examples run after the setup at the top of the page, in ``conductor/about/problems.md``. The setup makes a
``registry`` of small nodes: ``upper``, ``greet``, ``count``, ``words``, ``join``, ``route``, ``approve``, ``who``,
``pick`` and ``contacts``. It also makes one helper for each section:

- ``problems(record)`` compiles a graph record and returns its problems. It uses ``registry`` unless it is given
  another.
- ``refused(call)`` returns the problems a refused call raises.
- ``why_it_stopped(run)`` returns the problem a run stopped on.

An example's last line calls its section's helper. A test checks that the example gives problems of its own kind
and no other.
"""

from __future__ import annotations

from .call_problems import CALL_PROBLEMS
from .compile_problems import COMPILE_PROBLEMS
from .run_problems import RUN_PROBLEMS
from .type import ProblemCatalogue

#: Every kind of problem conductor's core reports.
PROBLEMS = ProblemCatalogue.joined(COMPILE_PROBLEMS, CALL_PROBLEMS, RUN_PROBLEMS)
