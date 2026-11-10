# ``Table[R]`` to a type checker: a Polars frame that keeps ``R``, the row type naming its columns.
# A node calls frame methods on a ``Table[R]`` input with no narrowing, and returns a plain ``polars.DataFrame``
# where ``Table[R]`` is declared. ``_RowsOf[R]`` exists only so the checker shows ``R`` and tells two row types apart.
from typing import Any

import polars

class _RowsOf[R](polars.DataFrame): ...

type Table[R = Any] = polars.DataFrame | _RowsOf[R]
