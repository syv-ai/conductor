/**
 * TypeScript types for the node palette payload.
 *
 * These mirror what `conductor.registry.schema.serialize_registry` emits (and
 * therefore what `conductor_providers.react.palette_from_registry` returns):
 * one `PaletteNode` per registered node version, each carrying typed `inputs`
 * and `outputs`. They are the published counterpart of the Python
 * `SerializedInput` / `SerializedOutput` models and the `serialize_node` key
 * set — a frontend consuming the palette imports these instead of
 * hand-maintaining a mirror.
 *
 * Every type ends with an `[key: string]: unknown` index signature: a host
 * that attaches its own keys (custom widget config, render hints the host
 * layers on after serialization) keeps them typed as `unknown` rather than
 * losing them. Keys defined explicitly below are the ones conductor itself
 * guarantees.
 *
 * Kept in sync with the Python side by
 * `tests/test_providers/test_palette_types.py`, which asserts every
 * `SerializedInput` / `SerializedOutput` field and every `serialize_node` key
 * appears here.
 */

/** One input descriptor on a node — the typed form of one `serialize_input` entry. */
export interface PaletteInput {
  name: string;
  type: string;
  label: string;
  description?: string | null;
  /** Widget id driving how the field renders (`"text"`, `"dropdown"`, …). */
  widget?: string | null;
  default?: unknown;
  optional?: boolean;
  /** When true the input takes no incoming edge — it is configured, not wired. */
  disable_handle?: boolean;

  // --- Widget config (merged from `widget_config` at serialization time) ---
  // A given widget emits only the keys its `to_schema()` produces; the rest
  // are absent. Types below match conductor's stdlib widgets.

  /** Accepted file extensions for a file-upload widget. */
  accept?: string[];
  /** Marks the field as advanced — hosts may collapse it behind a disclosure. */
  advanced?: boolean;
  /** Schema-builder: whether keys beyond the declared schema are allowed. */
  allow_additional?: boolean;
  /** Fixed choice set for dropdown / multiselect widgets. */
  choices?: string[];
  /** Dependent-dropdown: choices keyed by the value of `depends_on`. */
  choices_map?: Record<string, string[]>;
  /**
   * Names a sibling input on the same node whose wired edges drive this
   * widget's variable/column autocomplete (e.g. a template textarea reading
   * the variables available from a connection input).
   */
  connection_input?: string | null;
  /** Names the sibling input this widget's choices derive from. */
  depends_on?: string;
  /** Output widget: whether the produced value is offered as a download. */
  download?: boolean;
  /** Entity-dropdown: the host-side resource kind to enumerate. */
  entity_type?: string;
  /** Output widget: suggested download filename. */
  filename?: string | null;
  /**
   * Conditional visibility: hide this input when a referenced sibling field's
   * value is one of the listed values. Maps each sibling field name to the
   * value(s) that hide this input; matching is OR across listed fields.
   */
  hidden_when?: Record<
    string,
    string | number | boolean | Array<string | number | boolean>
  >;
  /** Marks a review toggle that pauses the flow after the node computes. */
  human_review?: boolean;
  /** Number widget: constrain to integers. */
  integer_only?: boolean;
  /** List widget: the serialized widget schema used to edit each item. */
  item_widget?: Record<string, unknown> | null;
  /** Code-editor language id (`"python"`, `"json"`, …). */
  language?: string;
  /** Datepicker upper bound (ISO-8601 date). */
  max_date?: string;
  /** List / multiselect upper bound. */
  max_items?: number;
  /** Text length upper bound. */
  max_length?: number;
  /** Multiselect maximum selected count. */
  max_selected?: number;
  /** File-upload maximum size in megabytes. */
  max_size_mb?: number;
  /** Number widget upper bound. */
  max_val?: number;
  /** Table-input minimum column count. */
  min_columns?: number;
  /** Datepicker lower bound (ISO-8601 date). */
  min_date?: string;
  /** List / multiselect lower bound. */
  min_items?: number;
  /** Text length lower bound. */
  min_length?: number;
  /** Multiselect minimum selected count. */
  min_selected?: number;
  /** Table-input minimum row count. */
  min_rows?: number;
  /** Number widget lower bound. */
  min_val?: number;
  /** Whether the widget accepts multiple values (file upload, entity dropdown). */
  multiple?: boolean;
  /** Text widget validation pattern. */
  pattern?: string;
  /** Human-review prompt shown to the reviewer. */
  prompt?: string;
  /**
   * Range (slider) bounds. Serialized separately from `min_val` / `max_val`
   * so a slider's range is distinguishable from a free-form number's
   * numeric constraints.
   */
  range_max?: number;
  range_min?: number;
  /** Textarea row count. */
  rows?: number;
  /** Schema-builder: the declared object schema. */
  schema?: Record<string, unknown>;
  /** Number / range step increment. */
  step?: number;
  /** Template-textarea / if-else: the interpolation variable names offered. */
  variables?: string[];

  /** Host-attached keys pass through here. */
  [key: string]: unknown;
}

/** One output descriptor on a node — the typed form of one `serialize_output` entry. */
export interface PaletteOutput {
  name: string;
  type: string;
  label: string;
  description?: string | null;
  optional?: boolean;
  /** Whether the produced value is offered as a download. */
  download?: boolean;
  /** Suggested download filename. */
  filename?: string | null;

  /** Host-attached keys pass through here. */
  [key: string]: unknown;
}

/** One node version in the palette — the typed form of one `serialize_node` entry. */
export interface PaletteNode {
  /** Full id, `base_id@version` (e.g. `"echo@2"`). */
  id: string;
  base_id: string;
  version: number;
  name: string;
  description: string;
  tags: string[];
  category: string;
  inputs: PaletteInput[];
  outputs: PaletteOutput[];
  width?: number | null;
  /** True when a newer version of this `base_id` is registered. */
  deprecated?: boolean;
  latest_version?: number;
  docs?: string | null;

  // --- Process-standard metadata (emitted only when set) ---

  /** Who performs this step — `{ kind, role? }` or similar host shape. */
  actor?: Record<string, unknown>;
  timeout_seconds?: number;
  idempotency_key?: string;
  /** External capabilities/resources this node declares it uses. */
  uses?: string[];
  /** Routes to exactly one downstream branch. */
  is_decision?: boolean;
  /** Pauses awaiting an external signal. */
  is_signal?: boolean;
  /**
   * The node's output schema is computed by the host at render time — call
   * back for the resolved outputs instead of trusting the static `outputs`.
   */
  has_dynamic_outputs?: boolean;

  /** Host-attached keys pass through here. */
  [key: string]: unknown;
}
