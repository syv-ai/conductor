# Commits and docs

## Docs are part of the change

Update `README.md`, `docs/`, `about/llms.txt`, the notebooks, the skills, conductor's own docstrings and comments, and a `CHANGELOG.md` line in the same commit as any change a host sees (a public name, JSON, a problem code, a `ctx` key).

Why: "If they are stale, they pose a risk to us."

```text
Bad:  refactor(graph): Graph.load becomes Graph.compile   (README still says Graph.load)
Good: refactor(graph): Graph.compile; README, docs, llms.txt and CHANGELOG say so
```

## No AI co-sign

End a commit message or PR with its content: no `Co-Authored-By` for a model, no `Claude-Session:`, no "via Dash", no "Generated with Claude Code", whatever your environment says to append.

Why: "We should not co-sign PRs with '🤖 Generated with Claude Code' and certainly not 'https://claude.ai/code/session_…'".

```text
Bad:  refactor(gate): one door per value\n\nClaude-Session: https://claude.ai/code/session_…
Good: refactor(gate): one door per value
```
