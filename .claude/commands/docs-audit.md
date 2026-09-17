---
description: Compare recent commits against project docs and propose edits (no commits)
---

Run a documentation audit against the recent history.

## How many commits

Look at the last **10** commits by default. If the user passed a number as the argument (e.g. `/docs-audit 30`), use that count instead. If the argument is `since-release`, compare against everything since the most recent git tag.

## Procedure

1. **Read the commit history.** Run `git log --oneline -<N>` where N is the commit count from above. For each commit, inspect `git show --stat <sha>` to see which files changed.

2. **Identify commits that could affect docs.** A commit is relevant if it:
   - Changes code under `packages/conductor/src/conductor/` (excluding `__pycache__`)
   - Adds or modifies a feature listed in `README.md` or `AGENTS.md`
   - Bumps a counted thing (test count, example count, file count) already cited in docs
   - Renames, splits, or removes a public API, module, or field

   Pure test changes, CI tweaks, dependency bumps, typo fixes, and internal refactors that don't touch public surface area can be skipped.

3. **Read every user-facing doc:**
   - `AGENTS.md`
   - `README.md`
   - `packages/conductor/src/conductor/about/llms.txt`
   - `docs/index.md`, `docs/OVERVIEW.md`, `docs/widgets.md`
   - `skills/add-node/SKILL.md`, `skills/create-graph/SKILL.md`

4. **Cross-reference.** For each relevant commit, ask:
   - Does the feature appear in the "Features" / "Highlights" lists?
   - Is the new API surface present in the API reference sections of `packages/conductor/src/conductor/about/llms.txt`?
   - Are data-model changes (new `GraphNode` fields, new questions on `CompiledGraph` / `CompiledNode` / `CompiledField`, new error types or cause codes, new problem codes, new events) reflected?
   - Have any cited counts drifted? (run `uv run pytest tests/ --collect-only -q 2>&1 | tail -1` for the authoritative test count, and `ls examples/*.ipynb | wc -l` for the notebook count)
   - Does the `## Feature Map` table in `packages/conductor/src/conductor/about/llms.txt` cover the new capability?
   - For changes that altered default behavior, does every doc above that describes the behavior still describe what the code does? Run a snippet you are unsure of rather than trusting it.

5. **Apply edits in place** to the files listed in step 3. Keep edits minimal and factual — describe what *exists now*, don't editorialize or add marketing language. Prefer tightening existing wording over adding new sections; add a new section only if a significant feature is missing entirely.

6. **Do not commit or push.** End your turn with a short bullet list of what you changed (or "no changes needed"). The user reviews via `git diff`, then decides whether to commit.

## Constraints

- **Touch only the files listed in step 3.** Do not edit code, tests or examples; a notebook that teaches removed API is flagged in the summary, not rewritten.
- If a commit message describes a feature that isn't actually in the code (wrong message), trust the code and mention the discrepancy in your summary — don't write docs for things that don't exist.
- Conductor's word is graph: a doc about the library never says "flow" (a host's word), except "data flow" and "control flow" as plain English. `tests/test_core/test_standalone.py` holds the shipped source to the same rule.
- If the pre-commit hook is active and you somehow reach for `git commit`, abort. This command is read-heavy and edit-heavy, but never committal.

## Output

At the end, print a summary like:

```
Changes applied:
- AGENTS.md: bumped test count 160 → 171 (commit abc123 added about module tests)
- packages/conductor/src/conductor/about/llms.txt: added CompiledNode.embedded_in to the compilation section
- README.md: no changes

Flagged but not changed:
- Commit def456 mentions a "batch mode" feature but no corresponding code exists.
```

If nothing needs changing, say so explicitly — "No drift detected across N commits" is a valid outcome.

$ARGUMENTS
