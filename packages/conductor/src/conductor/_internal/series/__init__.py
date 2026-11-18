"""Series: a list of values that a graph works through one item at a time.

A node that returns ``Series[T]`` gives a list. A node wired to it that takes a plain ``T`` is called once for
each item, and its outputs become a Series in turn. A node that takes ``Series[T]`` gets the whole list::

    @node("words", title="Words", description="Splits text into words.")
    def words(text: str) -> Series[str]:
        return text.split()

    @node("shout", title="Shout", description="Writes a word in capitals.")
    def shout(word: str) -> str:
        return word.upper()

Wired ``words`` to ``shout``, the text ``"hi there"`` calls ``shout`` twice. This package has the hint and how
a run stores a Series. The full rules are in the reference: ``python -m conductor.about one-or-many``.
"""

from .stored import Records, SeriesItems, StoredSeries
from .type import Series, SeriesType

__all__ = [
    "Records",
    "Series",
    "SeriesItems",
    "SeriesType",
    "StoredSeries",
]
