"""The problems conductor's core writes, one function each.

``PROBLEMS.problem`` says how a function's signature is read. A caller makes a problem by calling the function:
``core_problems.not_a_chain(after_id, before=before_id, inputs=open_inputs, outputs=outputs)``.
"""

from __future__ import annotations

from conductor._internal.problem import ProblemCatalogue

#: The kinds of problem conductor's core writes.
PROBLEMS = ProblemCatalogue()
