# Illustrated pages

Pages that show how conductor works, each with diagrams, examples you can step through, and the output the examples print. Each is one HTML file in `docs/explainers/`, so it also opens straight from disk; the problems page's catalogue and the standard nodes' palette are written in from the code when the docs are built (`docs/hooks/explainers.py`). A test runs every example on them and checks the output each page shows (`tests/test_core/test_explainer_samples_run.py`). Read them in order, or start with the one you need.

1. [Overview](explainers/overview.html): the map of the packages, one run from start to finish, and why conductor is built the way it is.
2. [A node and its types](explainers/nodes-and-types.html): what `@node` reads from a function, `Param`, the four shapes of output, `@host_type` and the hooks.
3. [The graph record and compiling](explainers/graph-record-and-compiling.html): the JSON record of a graph, and what `Graph.compile` checks and reports.
4. [Series and tables](explainers/series-and-tables.html): how `Series[T]` and `Table[R]` decide whether a node is called once or once per item.
5. [Asking a person](explainers/asking-a-person.html): a node that raises `Ask`, the questions a host is shown, answering and declining.
6. [Subgraphs](explainers/subgraphs.html): a graph record used as one node of another graph.
7. [Saved runs and reuse](explainers/saved-runs-and-reuse.html): a run as JSON, read back, and run again with the calls it can reuse.
8. [Running and streaming](explainers/running-and-streaming.html): `resume()`, `stream()`, the scheduler, outcomes, status and events.
9. [Problems](explainers/problems.html): how conductor reports what went wrong, and every kind of problem.
10. [The standard nodes](explainers/standard-nodes.html): the ready-made nodes of `syv-conductor-nodes`.
11. [Serving runs](explainers/serving-runs.html): `syv-conductor-providers`, which keeps runs in a host's store and serves them over HTTP.
