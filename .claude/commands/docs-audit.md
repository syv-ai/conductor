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
   - Adds or modifies a feature listed in `README.md` or `CLAUDE.md`
   - Bumps a counted thing (test count, example count, file count) already cited in docs
   - Renames, splits, or removes a public API, module, or field

   Pure test changes, CI tweaks, dependency bumps, typo fixes, and internal refactors that don't touch public surface area can be skipped.

3. **Read every user-facing doc:**
   - `CLAUDE.md`
   - `README.md`
   - `docs/llms.txt`
   - `docs/shared-references.md`
   - `docs/index.md`

4. **Cross-reference.** For each relevant commit, ask:
   - Does the feature appear in the "Features" / "Highlights" lists?
   - Is the new API surface present in the API reference sections of `docs/llms.txt`?
   - Are data-model changes (new `GraphNode` fields, new `CompiledGraph` fields, new error types, new events) reflected?
   - Have any cited counts drifted? (run `uv run pytest tests/ --collect-only -q 2>&1 | tail -1` for the authoritative test count, and `ls examples/*.ipynb | wc -l` for the notebook count)
   - Does the `## Feature Map` table in `docs/llms.txt` cover the new capability?
   - For changes that altered default behavior, does the design spec for that feature (e.g. `docs/shared-references.md` for shared references) still describe the current behavior?

5. **Apply edits in place** to the files listed in step 3. Keep edits minimal and factual — describe what *exists now*, don't editorialize or add marketing language. Prefer tightening existing wording over adding new sections; add a new section only if a significant feature is missing entirely.

6. **Do not commit or push.** End your turn with a short bullet list of what you changed (or "no changes needed"). The user reviews via `git diff`, then decides whether to commit.

## Constraints

- **Touch only these files:** `CLAUDE.md`, `README.md`, `docs/llms.txt`, `docs/shared-references.md`, `docs/index.md`. Do not edit code, tests, examples, or the demo.
- If a commit message describes a feature that isn't actually in the code (wrong message), trust the code and mention the discrepancy in your summary — don't write docs for things that don't exist.
- If the audit would require changes to a design doc (§ of `docs/shared-references.md`, etc.) that has ongoing implications, flag it in the summary instead of silently rewriting — the user may want to discuss before the spec is mutated.
- If the pre-commit hook is active and you somehow reach for `git commit`, abort. This command is read-heavy and edit-heavy, but never committal.

## Output

At the end, print a summary like:

```
Changes applied:
- CLAUDE.md: bumped test count 160 → 171 (commit abc123 added about module tests)
- docs/llms.txt: added conductor.graph.shared_refs to API reference
- README.md: no changes

Flagged but not changed:
- Commit def456 mentions a "batch mode" feature but no corresponding code exists.
```

If nothing needs changing, say so explicitly — "No drift detected across N commits" is a valid outcome.

$ARGUMENTS
