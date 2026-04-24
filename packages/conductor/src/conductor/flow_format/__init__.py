"""Author-facing YAML / JSON flow format.

The engine is Python-API-first: you instantiate :class:`GraphNode` and
:class:`GraphEdge` directly. That's great for programmatic flow building
but terrible for stakeholders who want to read a flow like a spec,
or for version-control diffs.

This module converts between a canonical YAML/JSON dict shape and the
rich :class:`conductor.Flow` object. It's completely optional — nothing
in the engine imports it.

The shape::

    id: my-flow
    version: 1
    name: Onboarding
    on_error_default: fail
    dependencies:
      - id: stripe
        kind: api
        config: {endpoint: https://api.stripe.com}
    triggers:
      - id: nightly
        kind: schedule
        config: {cron: "0 9 * * *", timezone: UTC}
    nodes:
      - id: charge
        type: stripe-charge@1
        data: {amount: 1000}
        compensation: refund
        actor: {kind: system, role: stripe}
      - id: refund
        type: stripe-refund@1
    edges:
      - id: e1
        source: charge
        target: refund
        when: "amount > 1000"
        priority: 10
"""

from conductor.flow_format.loader import (
    dump_flow,
    flow_to_dict,
    flow_to_yaml,
    load_flow,
    load_flow_from_path,
    yaml_to_flow,
)

__all__ = [
    "load_flow",
    "load_flow_from_path",
    "yaml_to_flow",
    "flow_to_yaml",
    "dump_flow",
    "flow_to_dict",
]
