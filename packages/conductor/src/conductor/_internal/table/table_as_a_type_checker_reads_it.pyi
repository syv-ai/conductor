# What a type checker sees for ``Table[R]``: a Polars frame that remembers its row type ``R``.
# So a node can call frame methods on a ``Table[R]`` input without narrowing it first, and can return a plain
# ``polars.DataFrame`` where it declares ``Table[R]``. ``_RowsOf[R]`` is there only so the checker shows ``R``
# and can tell ``Table[Invoice]`` from ``Table[Customer]``.
from typing import Any

import polars

class _RowsOf[R](polars.DataFrame): ...

type Table[R = Any] = polars.DataFrame | _RowsOf[R]
