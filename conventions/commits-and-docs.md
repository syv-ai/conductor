# Commits and docs

## Docs are part of the change

Update `README.md`, `docs/`, `about/llms.txt`, the notebooks, the skills, conductor's own docstrings and comments, and a `CHANGELOG.md` line in the same commit as any change a host sees (a public name, JSON, a problem code, a `ctx` key).

Why: stale docs teach hosts and agents the wrong thing, and they act on it.

```text
Bad:  refactor(graph): Graph.load becomes Graph.compile   (README still says Graph.load)
Good: refactor(graph): Graph.compile; README, docs, llms.txt and CHANGELOG say so
```
