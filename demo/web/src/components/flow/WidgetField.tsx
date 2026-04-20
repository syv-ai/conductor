"use client";
import * as React from "react";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/utils";
import type { NodeParam } from "@/lib/types";

interface Props {
  param: NodeParam;
  value: unknown;
  disabled: boolean; // true when an edge or a consume binding fills this input
  onChange: (value: unknown) => void;
}

export function WidgetField({ param, value, disabled, onChange }: Props) {
  const ring = cn("w-full", disabled && "opacity-40 pointer-events-none");

  switch (param.widget) {
    case "textarea":
    case "template-textarea":
    case "code-editor":
      return (
        <Textarea
          rows={param.rows ?? 3}
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className={ring}
        />
      );

    case "dropdown":
      return (
        <Select
          value={value == null ? "" : String(value)}
          onValueChange={onChange}
          disabled={disabled}
        >
          <SelectTrigger className={ring}>
            <SelectValue placeholder="Select…" />
          </SelectTrigger>
          <SelectContent>
            {(param.choices ?? []).map((c) => (
              <SelectItem key={c} value={c}>
                {c}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      );

    case "range":
    case "number": {
      const num = typeof value === "number" ? value : Number(value ?? 0);
      const min = param.min_val ?? -1000;
      const max = param.max_val ?? 1000;
      const step = param.step ?? (param.integer_only ? 1 : 0.1);
      return (
        <div className={cn("flex items-center gap-2", disabled && "opacity-40 pointer-events-none")}>
          <Slider
            min={min}
            max={max}
            step={step}
            value={[num]}
            onValueChange={(v) => onChange(v[0])}
            disabled={disabled}
            className="flex-1"
          />
          <Input
            type="number"
            min={min}
            max={max}
            step={step}
            value={num}
            onChange={(e) => onChange(Number(e.target.value))}
            disabled={disabled}
            className="h-7 w-16 text-xs"
          />
        </div>
      );
    }

    case "checkbox":
    case "switch":
      return (
        <Switch
          checked={Boolean(value)}
          onCheckedChange={onChange}
          disabled={disabled}
        />
      );

    case "connection-list":
      return (
        <div className="rounded-md border border-dashed bg-muted/40 px-2 py-1 text-[11px] text-muted-foreground">
          Connect multiple edges to this input.
        </div>
      );

    case "text":
    default:
      return (
        <Input
          value={String(value ?? "")}
          onChange={(e) => onChange(e.target.value)}
          disabled={disabled}
          className={ring}
        />
      );
  }
}
