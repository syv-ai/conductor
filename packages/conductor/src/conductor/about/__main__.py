"""CLI entry point: ``python -m conductor.about [section]``."""

from __future__ import annotations

import sys

from conductor.about import get_content, get_section, list_sections


_USAGE = """\
Usage: python -m conductor.about [section]

  (no args)              print the full reference text
  sections               list available section slugs
  <slug>                 print the matching section (prefix/substring ok)
  -h, --help             show this message
"""


def main(argv: list[str]) -> int:
    if not argv:
        sys.stdout.write(get_content())
        return 0

    arg = argv[0]

    if arg in ("-h", "--help"):
        sys.stdout.write(_USAGE)
        return 0

    if arg == "sections":
        for slug in list_sections():
            print(slug)
        return 0

    section = get_section(arg)
    if section is None:
        print(f"No section matching '{arg}'. Available:", file=sys.stderr)
        for slug in list_sections():
            print(f"  {slug}", file=sys.stderr)
        return 1

    sys.stdout.write(section)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
